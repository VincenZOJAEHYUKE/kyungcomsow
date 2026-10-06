import base64
import csv
import json
import os
import re
import threading
import time
import requests
from datetime import datetime
import textwrap
from PIL import Image, ImageDraw, ImageFont

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from ultralytics import YOLO
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()
# CORS 설정 (HTML 대시보드와 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# 설정 및 초기화
# ---------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=30_000),  # VLM 응답 대기 30초 (ms 단위)
)
GEMINI_MODEL = "gemini-3.5-flash-lite"

# YOLO 모델 로드 (같은 폴더의 best.pt 활용)
model = YOLO("best.pt")
yolo_lock = threading.Lock()  # YOLO 추론은 한 번에 하나만

# --- 검출 파라미터 ---
DET_CONF = 0.75  # 검출 신뢰도 (규격서 가안은 0.6, [임계값실험] 후 확정)
MAX_AREA_RATIO = 0.3  # 화면의 30% 이상인 박스는 오탐지(얼굴 등)로 무시. 차가 크게 잡히면 올릴 것
MIN_AREA_RATIO = 0.02  # 너무 작은 박스는 캡처 대상에서 제외
EDGE_MARGIN = 0.02  # 박스가 화면 가장자리 2% 안에 걸치면 잘린 차량으로 본다

# --- 자동 캡처 파라미터 (규격서 v0 가안) ---
COOLDOWN_SEC = 4.0  # ★ 마지막 캡처 후 이 시간 동안은 캡처 차단 (3~5초)
N_STABLE = 5  # 연속 유효 검출 + 박스 안 움직임 프레임 수
IOU_MIN = 0.7  # 연속 두 프레임 박스 겹침 기준
N_ABSENT = 10  # 연속 미검출이면 차량이 나간 것으로 본다
ANALYZE_TIMEOUT_SEC = 40.0  # 분석이 이 시간 넘게 안 끝나면 강제로 locked (시스템 정지 방지)
CROP_PADDING = 0.10  # VLM에 보낼 크롭 여백 10%

# --- 저장 경로 ---
LOG_PATH = "capture_log.csv"
CAPTURE_DIR = "captures"  # 변수 선언을 먼저 위로 올립니다.
app.mount("/captures", StaticFiles(directory=CAPTURE_DIR), name="captures") # 선언된 변수 사용
os.makedirs(CAPTURE_DIR, exist_ok=True)
# 상태별 하위 폴더 생성
os.makedirs(os.path.join(CAPTURE_DIR, "정상"), exist_ok=True)
os.makedirs(os.path.join(CAPTURE_DIR, "수리"), exist_ok=True)
os.makedirs(os.path.join(CAPTURE_DIR, "폐기"), exist_ok=True)
os.makedirs(os.path.join(CAPTURE_DIR, "미분류"), exist_ok=True)

SYSTEM_PROMPT = """당신은 열악한 제조 공정(저조도 및 고탁도 환경)에서 복합 재질 부품의 불량을 탐지하고 분류하는 자율 AI 에이전트입니다.
전달받은 이미지는 시각 지능(YOLO)을 통해 1차적으로 탐지 및 크롭된 객체 이미지입니다.

[판단 및 분류 규칙]
1. 복합 재질(플라스틱, 유리, 고무)의 상태와 불량 심각도를 정밀하게 분석하세요.
2. 재질별 세부 판정 기준:
   - 유리: 심하게 금이 가거나 파손된 경우 "폐기" 판정. 표면의 가벼운 탁도나 얼룩은 "수리(세척)".
   - 플라스틱 (메인 프레임 등): 구조적 찌그러짐이나 주요 부품 파손 및 이탈(문짝 없음)시 "폐기". 가벼운 스크래치나 도장 벗겨짐은 "폐기".
   - 투명 플라스틱 창문 누락시 "수리".
   - 고무 (타이어 등): 찢어짐이나 이탈이 발생한 구조적 결함은 "수리".
3. 정상적인 부품의 경우 "정상"으로 판정하여 재사용(Pass) 하도록 합니다.
4. 객체를 식별할 수 없을 정도로 노이즈가 심하거나 배경만 있는 경우 "판정 불가"로 출력하세요.
5. 반드시 아래 JSON 형식으로만 답하세요.

{
  "status": "정상" 또는 "수리" 또는 "폐기" 또는 "판정 불가",
  "confidence": 0.95,
  "reason": "상세 판정 사유 (예: 고탁도 환경 이미지 분석 결과, 플라스틱 프레임 우측에 심한 찌그러짐이 확인되어 폐기 조치합니다.)",
  "actuator_action": "하드웨어 제어 명령 (예: 폐기 라인 분류 스위치 ON)"
}"""


# ---------------------------------------------------------------
# 공통 함수: Gemini 판정 / 박스 계산
# ---------------------------------------------------------------
def ask_gemini(image_bytes, criteria=None):
    """크롭된 JPEG 바이트를 Gemini로 보내 판정 JSON(dict)을 받는다."""
    prompt = "검수 기준을 참고하여 자동차의 상태를 판정해주세요."
    if criteria:
        prompt += f"\n기준: {criteria}"

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            max_output_tokens=1024,
        ),
    )
    text = re.sub(r"^```(?:json)?|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


def iou(a, b):
    """두 박스 [x1,y1,x2,y2]의 겹침 비율(IoU)"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pick_car(result, w, h):
    """YOLO 결과에서 대상 차량 1대를 고른다. 없으면 None.
    박스는 0~1 비율 [x1,y1,x2,y2]로 반환한다 (규격서 3-1)."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

    candidates = [i for i, a in enumerate(areas) if a < w * h * MAX_AREA_RATIO]
    if not candidates:
        return None
    i = max(candidates, key=lambda k: areas[k])
    x1, y1, x2, y2 = xyxy[i]
    return {
        "box": [float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)],
        "conf": float(confs[i]),
        "area_ratio": float(areas[i] / (w * h)),
    }


def check_valid(det):
    """캡처 대상으로 쓸 수 있는 '유효한 검출'인지 확인. (유효 여부, 이유)"""
    if det is None:
        return False, "no_car"
    x1, y1, x2, y2 = det["box"]
    if x1 < EDGE_MARGIN or y1 < EDGE_MARGIN or x2 > 1 - EDGE_MARGIN or y2 > 1 - EDGE_MARGIN:
        return False, "edge"  # 가장자리에 걸려 잘린 차량
    if det["area_ratio"] < MIN_AREA_RATIO:
        return False, "too_small"
    return True, "ok"


def crop_vehicle(frame, box):
    """0~1 비율 박스를 여백 포함해 원본 해상도로 크롭"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    pw, ph = (x2 - x1) * CROP_PADDING, (y2 - y1) * CROP_PADDING
    return frame[
        max(0, int(y1 - ph)) : min(h, int(y2 + ph)),
        max(0, int(x1 - pw)) : min(w, int(x2 + pw)),
    ]


# ---- CaptureController START ----
class CaptureController:
    """언제 캡처할지 결정하는 상태 머신 (규격서 2번).

    idle      차량 없음
    tracking  차량이 보임, 안정되기를 기다림
    analyzing VLM 판정 중 (새 캡처 금지)
    locked    판정 끝, 차량이 나가기를 기다림 (같은 차량 재캡처 방지)

    같은 차량이 계속 찍히는 것은 두 겹으로 막는다.
      1) 상태: 한 번 캡처하면 차량이 나갈 때까지(locked) 캡처하지 않는다.
      2) 쿨다운: last_capture_time으로부터 COOLDOWN_SEC 안에는 어떤 경우에도 캡처하지 않는다.
    """

    def __init__(self, start_id=1):
        self.lock = threading.Lock()
        self.state = "idle"
        self.stable = 0
        self.absent = 0
        self.prev_box = None
        self.last_capture_time = float("-inf")  # ★ 마지막 캡처 시각
        self.analyze_start = 0.0
        self.next_id = start_id

    def update(self, valid, box, now, why):
        """프레임 1장 처리. (캡처 여부, 이유, capture_id) 반환"""
        with self.lock:
            # 분석이 안 끝나고 멈춘 경우를 대비한 안전장치
            if self.state == "analyzing" and now - self.analyze_start > ANALYZE_TIMEOUT_SEC:
                self.state = "locked"

            if valid:
                self.absent = 0
            else:
                self.absent += 1
                self.stable = 0

            captured, capture_id = False, None
            reason = "ok" if valid else why

            if self.state == "idle":
                if valid:
                    self.state, self.stable = "tracking", 1
                    reason = "tracking_start"

            elif self.state == "tracking":
                if valid:
                    if self.prev_box is not None and iou(box, self.prev_box) >= IOU_MIN:
                        self.stable += 1
                    else:
                        self.stable = 1
                        reason = "unstable"

                    if self.stable >= N_STABLE:
                        if now - self.last_capture_time >= COOLDOWN_SEC:
                            captured, capture_id = True, self.next_id
                            self.next_id += 1
                            self.state = "analyzing"
                            self.last_capture_time = now
                            self.analyze_start = now
                            reason = "captured"
                        else:
                            reason = "cooldown"
                    elif reason == "ok":
                        reason = "waiting_stable"

                if self.absent >= N_ABSENT:
                    self.state = "idle"

            elif self.state == "analyzing":
                reason = "analyzing"

            elif self.state == "locked":
                reason = "locked"
                if self.absent >= N_ABSENT:
                    self.state = "idle"  # 차량이 나가야 다음 차량을 받는다

            self.prev_box = box if valid else None
            return captured, reason, capture_id

    def on_analysis_done(self):
        """판정이 끝나거나 실패하면 locked로. 같은 차량은 나갈 때까지 다시 찍지 않는다."""
        with self.lock:
            if self.state == "analyzing":
                self.state = "locked"


# ---- CaptureController END ----


def _next_capture_id():
    """재시작해도 기존 파일을 덮어쓰지 않도록 다음 번호를 계산"""
    nums = [int(m.group(1)) for f in os.listdir(CAPTURE_DIR) if (m := re.fullmatch(r"(\d+)\.jpg", f))]
    return max(nums, default=0) + 1


controller = CaptureController(start_id=_next_capture_id())
latest_result = {"capture_id": None, "status": "none", "lamp": "none", "log": "아직 판정 결과가 없습니다."}
result_lock = threading.Lock()
log_lock = threading.Lock()

LOG_FIELDS = [
    "ts", "frame_id", "detected", "confidence", "state",
    "captured", "capture_id", "reason", "detect_ms", "vlm_ms",
]


def write_log(**row):
    """capture_log.csv에 한 줄 기록 (규격서 5번). 로그 실패가 서버를 멈추지 않게 한다."""
    try:
        with log_lock:
            is_new = not os.path.exists(LOG_PATH)
            with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                if is_new:
                    writer.writeheader()
                writer.writerow({"ts": int(time.time() * 1000), **row})
    except Exception as e:
        print(">> 로그 기록 실패:", e)


def save_capture(capture_id, jpeg_bytes):
    """캡처한 원본 프레임을 000007.jpg 형태로 저장 (cv2.imwrite는 한글 경로에서 실패하므로 직접 씀)"""
    path = os.path.join(CAPTURE_DIR, f"{capture_id:06d}.jpg")
    with open(path, "wb") as f:
        f.write(jpeg_bytes)
    return path


# 판정 -> (영문 status, 호출할 API, 램프 색)
STATUS_MAP = {
    "정상": ("pass", "/api/pass", "green"),
    "수리": ("repair", "/api/repair", "yellow"),
    "폐기": ("discard", "/api/discard", "red"),
}


def analyze_capture(capture_id, crop_bytes, t_capture):
    """캡처 직후 별도 스레드에서 VLM 판정 (5~6단계). 성공이든 실패든 끝나면 locked로 전환."""
    global latest_result
    try:
        t0 = time.time()
        r = ask_gemini(crop_bytes)
        vlm_ms = int((time.time() - t0) * 1000)

        label = r.get("status", "판정 불가")
        status, api, lamp = STATUS_MAP.get(label, ("unknown", None, "none"))
        confidence = float(r.get("confidence", 0))
        actuator_action = r.get("actuator_action", "")

        # 가상 하드웨어 API 호출 (자율 실행)
        if api:
            try:
                hw_url = f"http://127.0.0.1:8080{api}" 
                requests.post(hw_url, json={"action": actuator_action}, timeout=1.0)
                print(f">> [Actuator] {status} 제어 명령 전송 완료: {actuator_action}")
            except Exception as req_err:
                print(f">> [Actuator] 하드웨어 제어 통신 실패: {req_err}")

        # --- 판정 결과 문구에 따른 유연한 타겟 폴더 결정 및 이미지 텍스트 각인 ---
        if "정상" in label:
            target_folder = "정상"
        elif "수리" in label:
            target_folder = "수리"
        elif "폐기" in label or "불량" in label:
            target_folder = "폐기"
        else:
            target_folder = "미분류"
        
        # 원본 파일 경로와 이동할 새 경로 설정
        original_filepath = os.path.join("captures", f"{capture_id:06d}.jpg")
        new_filepath = os.path.join("captures", target_folder, f"{capture_id:06d}.jpg")
        
        if os.path.exists(original_filepath):
            try:
                # 1. OpenCV로 원본 이미지 읽기 (한글 경로 문제 방지)
                img_array = np.fromfile(original_filepath, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # 2. 한글 출력을 위해 PIL 이미지 포맷으로 변환
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)

                # 윈도우 기본 폰트(맑은 고딕) 설정 (크기 조정)
                try:
                    font_title = ImageFont.truetype("malgun.ttf", 32)
                    font_body = ImageFont.truetype("malgun.ttf", 18)
                except IOError:
                    font_title = ImageFont.load_default()
                    font_body = ImageFont.load_default()

                # 판정 결과에 따른 텍스트 색상 설정 (RGB)
                if target_folder == "정상":
                    color = (50, 205, 50)  # 초록색
                elif target_folder == "수리":
                    color = (255, 165, 0)  # 주황색
                else:
                    color = (255, 69, 0)   # 빨간색

                # 텍스트 구성 및 줄바꿈 글자 수 축소 (width=32로 설정하여 화면 밖으로 나가는 것 방지)
                text_status = f"판정: {label} (신뢰도: {confidence:.2f})"
                reason_text = r.get("reason", "")
                text_reason = textwrap.fill(reason_text, width=32)

                # 가독성을 높이기 위해 검은색 그림자를 먼저 그리고 본 텍스트를 덮어씀
                draw.text((22, 22), text_status, font=font_title, fill=(0, 0, 0))
                draw.text((22, 68), text_reason, font=font_body, fill=(0, 0, 0))
                
                draw.text((20, 20), text_status, font=font_title, fill=color)
                draw.text((20, 66), text_reason, font=font_body, fill=(255, 255, 255))

                # 3. OpenCV 포맷으로 다시 변환 후 새 폴더에 저장
                img_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                _, encoded_img = cv2.imencode('.jpg', img_with_text)
                
                with open(new_filepath, mode='w+b') as f:
                    encoded_img.tofile(f)

                # 4. 처리가 끝난 원본 파일 삭제
                os.remove(original_filepath)
                print(f">> [File] 텍스트 각인 및 분류 완료: {target_folder} 폴더")
                
            except Exception as e:
                print(f">> [Error] 이미지 텍스트 추가 실패 (단순 이동 시도): {e}")
                os.rename(original_filepath, new_filepath)
        # ------------------------------------------------------------------------

        result = {
            "capture_id": capture_id,
            "status": status,
            "label": label,
            "api": api, 
            "lamp": lamp,
            "needs_review": confidence < 0.6,
            "confidence": confidence,
            "defects": [], 
            "log": r.get("reason", ""),
            "actuator_action": actuator_action,
            "image_url": f"/captures/{target_folder}/{capture_id:06d}.jpg",
            "timing_ms": {"vlm": vlm_ms, "total": int((time.time() - t_capture) * 1000)},
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        write_log(frame_id="", capture_id=capture_id, state="analyzing", reason="vlm_done", vlm_ms=vlm_ms)
        
    except Exception as e:
        print(f">> 판정 실패 (capture {capture_id}):", e)
        result = {
            "capture_id": capture_id,
            "status": "error",
            "lamp": "none",
            "log": f"판정 실패: {e}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        write_log(frame_id="", capture_id=capture_id, state="analyzing", reason="vlm_error")

    with result_lock:
        latest_result = result
    controller.on_analysis_done()

# ---------------------------------------------------------------
# 연속 프레임 파이프라인
# ---------------------------------------------------------------
@app.post("/api/frame")
def receive_frame(
    file: UploadFile = File(...), frame_id: int = Form(None), sent_at: str = Form(None)
):
    """프론트가 1초에 3~5장씩 보내는 프레임을 받아 YOLO -> 캡처 컨트롤러로 넘긴다.
    동기(def)로 선언해서 FastAPI가 스레드풀에서 돌리므로 추론 중에도 다른 요청이 막히지 않는다."""
    contents = file.file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "이미지를 읽을 수 없습니다."})

    # 이전 프레임을 아직 처리 중이면 이 프레임은 버린다 (지연이 쌓이지 않게)
    if not yolo_lock.acquire(blocking=False):
        write_log(frame_id=frame_id, state=controller.state, captured=False, reason="dropped")
        return {"frame_id": frame_id, "dropped": True, "state": controller.state}
    try:
        t0 = time.time()
        results = model(frame, conf=DET_CONF, verbose=False)
        detect_ms = int((time.time() - t0) * 1000)
    finally:
        yolo_lock.release()

    h, w = frame.shape[:2]
    det = pick_car(results[0], w, h)
    valid, why = check_valid(det)
    captured, reason, capture_id = controller.update(
        valid, det["box"] if valid else None, time.monotonic(), why
    )

    if captured:
        try:
            save_capture(capture_id, contents)  # 박스가 그려지기 전 원본 프레임 그대로 저장
            ok, buf = cv2.imencode(".jpg", crop_vehicle(frame, det["box"]), [cv2.IMWRITE_JPEG_QUALITY, 90])
            threading.Thread(
                target=analyze_capture, args=(capture_id, buf.tobytes(), time.time()), daemon=True
            ).start()
        except Exception as e:  # 저장 실패해도 analyzing에 갇히지 않게
            print(">> 캡처 처리 실패:", e)
            reason = "capture_error"
            controller.on_analysis_done()

    state = controller.state
    write_log(
        frame_id=frame_id,
        detected=det is not None,
        confidence=round(det["conf"], 3) if det else "",
        state=state,
        captured=captured,
        capture_id=capture_id or "",
        reason=reason,
        detect_ms=detect_ms,
    )

    return {
        "frame_id": frame_id,
        "detected": det is not None,
        "bbox": det["box"] if det else None,
        "confidence": det["conf"] if det else None,
        "state": state,
        "capture_triggered": captured,
        "reason": reason,
    }


@app.get("/api/result/latest")
def get_latest_result():
    """프론트가 1초마다 조회하는 최신 판정 결과 (규격서 3-2, 3-3)"""
    with result_lock:
        return latest_result


# ---------------------------------------------------------------
# 기존: 이미지 1장 업로드 판정 (백업/수동 테스트용)
# ---------------------------------------------------------------
@app.post("/evaluate-car")
async def evaluate_car(
    file: UploadFile = File(...), inspection_criteria: str = Form(None)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. YOLO 객체 탐지
        with yolo_lock:
            results = model(frame, conf=DET_CONF, verbose=False)
        boxes = results[0].boxes

        h, w = frame.shape[:2]
        image_area = h * w

        annotated_frame = frame.copy()  # 박스를 그릴 도화지 복사본

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

            # 화면 면적 30% 미만인 박스만 유효(valid) 처리 (얼굴 등 거대한 오탐지 제외)
            valid_indices = [i for i, area in enumerate(areas) if area < (image_area * MAX_AREA_RATIO)]

            if valid_indices:
                # 유효한 박스 중 가장 큰 것 선택
                best_idx = max(valid_indices, key=lambda i: areas[i])
                x1, y1, x2, y2 = map(int, xyxy[best_idx])
                conf = float(boxes.conf[best_idx])

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                text_label = f"Car {conf:.2f}"
                cv2.putText(annotated_frame, text_label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # 크롭 진행
                pw, ph = (x2 - x1) * 0.05, (y2 - y1) * 0.05
                crop = frame[
                    max(0, int(y1 - ph)) : min(h, int(y2 + ph)),
                    max(0, int(x1 - pw)) : min(w, int(x2 + pw)),
                ]
            else:
                crop = frame
        else:
            crop = frame

        # 박스가 그려진 이미지를 Base64 문자열로 변환하여 HTML로 전송 준비
        _, buffer = cv2.imencode(".jpg", annotated_frame)
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        # 3. 크롭된 이미지를 Gemini로 전송
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        result_json = ask_gemini(buf.tobytes(), inspection_criteria)

        # JSON 결과에 박스가 그려진 이미지 데이터를 함께 끼워넣음
        result_json["annotated_image"] = f"data:image/jpeg;base64,{annotated_b64}"

        return result_json

    except Exception as e:
        print(">> 서버 에러 발생:", str(e))
        return JSONResponse(status_code=500, content={"error": f"서버 내부 오류: {str(e)}"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
