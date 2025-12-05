import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"

import sys
import cv2
import numpy as np
import torch
import time
from datetime import datetime
from pathlib import Path
import threading
import requests
import base64
import json
import select
import socket
# 음성 인식용
import pyaudio
import wave
import speech_recognition as sr
import tempfile
import google.generativeai as genai

# 환경 변수 및 커널 설정
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
torch.backends.cuda.enable_flash_sdp = False
torch.backends.cuda.enable_mem_efficient_sdp = True
torch.backends.cuda.enable_math_sdp = True

# SAM2 관련 임포트
sys.path.append('/home/j/sam4/src/SAM2_streaming')
from sam2.build_sam import build_sam2_camera_predictor

# RAFT-Stereo 관련 임포트 및 경로 추가
sys.path.append('core')
from raft_stereo import RAFTStereo
from utils.utils import InputPadder

# ====== 사용자 파라미터/경로 ======
calib_data = np.load('/home/j/stereo/stereo_calibration_data.npz')
mtx_left = calib_data['mtx_left']
dist_left = calib_data['dist_left']
mtx_right = calib_data['mtx_right']
dist_right = calib_data['dist_right']
R = calib_data['R']
T = calib_data['T']

model_version = 'sam2.1'
model_size = 'tiny'
sam2_checkpoint = f"./checkpoints/{model_version}/{model_version}_hiera_{model_size}.pt"
model_cfg = f"{model_version}/{model_version}_hiera_{model_size[0]}.yaml"

DEVICE = 'cuda'
torch.cuda.empty_cache()

# RAFT-Stereo 설정
raft_args = type('', (), {})()
raft_args.restore_ckpt = 'models/raftstereo-middlebury.pth'
raft_args.output_directory = 'demo_output/example/test12'
raft_args.save_numpy = False
raft_args.mixed_precision = False
raft_args.valid_iters = 128  # 정밀 계산용 반복 횟수 (높음)
raft_args.hidden_dims = [128]*3
raft_args.corr_implementation = 'reg_cuda'
raft_args.shared_backbone = False
raft_args.corr_levels = 4
raft_args.corr_radius = 4
raft_args.n_downsample = 2
raft_args.context_norm = 'instance'
raft_args.slow_fast_gru = True
raft_args.n_gru_layers = 3

# V4L2 백엔드 사용
cap_left = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap_right = cv2.VideoCapture(4, cv2.CAP_V4L2)
cap_left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap_right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_left.set(cv2.CAP_PROP_FPS, 30)
cap_right.set(cv2.CAP_PROP_FPS, 30)

ret_left, frame_left = cap_left.read()
ret_right, frame_right = cap_right.read()

print(f"왼쪽 해상도: {cap_left.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

image_size = (frame_left.shape[1], frame_left.shape[0])
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_left, dist_left, mtx_right, dist_right,
    image_size, R.T, -T, alpha=-1, flags=cv2.CALIB_ZERO_DISPARITY
)
left_mapx, left_mapy = cv2.initUndistortRectifyMap(
    mtx_left, dist_left, R1, P1, image_size, cv2.CV_32FC1)
right_mapx, right_mapy = cv2.initUndistortRectifyMap(
    mtx_right, dist_right, R2, P2, image_size, cv2.CV_32FC1)

save_dir = raft_args.output_directory
os.makedirs(save_dir, exist_ok=True)

# --- [최적화] NumPy 이미지를 바로 GPU Tensor로 변환 ---
def frame_to_tensor(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

# --- 모델 로드 함수 ---
def load_raftstereo_model(raft_args, device):
    model = RAFTStereo(raft_args).to(device)
    
    if torch.cuda.is_available():
        checkpoint = torch.load(raft_args.restore_ckpt, map_location=torch.device('cuda'), weights_only=False)
    else:
        checkpoint = torch.load(raft_args.restore_ckpt, map_location=torch.device('cpu'), weights_only=False)
        
    new_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    
    filtered_checkpoint = {}
    for k, v in new_checkpoint.items():
        if 'running_mean' in k or 'running_var' in k:
            continue
        filtered_checkpoint[k] = v
        
    model.load_state_dict(filtered_checkpoint, strict=False)
    model.eval()
    return model

raftstereo_model = load_raftstereo_model(raft_args, DEVICE)

# --- [핵심] 실시간 깊이 맵 생성 함수 (속도 최적화) ---
def get_realtime_disparity(left_img, right_img, model=raftstereo_model, viz_iters=12):
    """
    실시간 시각화를 위해 적은 반복 횟수(viz_iters)로 disparity를 계산합니다.
    """
    with torch.no_grad():
        image1 = frame_to_tensor(left_img)
        image2 = frame_to_tensor(right_img)

        padder = InputPadder(image1.shape, divis_by=32)
        image1_pad, image2_pad = padder.pad(image1, image2)
        
        # test_mode=True, iters를 낮게 설정하여 고속 추론
        _, flow_up = model(image1_pad, image2_pad, iters=viz_iters, test_mode=True)
        
        disparity = -padder.unpad(flow_up).squeeze().cpu().numpy()
        
        # 시각화를 위한 정규화 (0~255)
        disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_MAGMA)
        
        return disp_color

# --- 정밀 좌표 계산 함수 ---
def calculate_centroid_optimized(left_img, right_img, mask_img, model=raftstereo_model):
    start = time.time()
    Q_flipped = Q.copy()
    Q_flipped[3,2] = -Q_flipped[3,2]

    with torch.no_grad():
        image1 = frame_to_tensor(left_img)
        image2 = frame_to_tensor(right_img)

        padder = InputPadder(image1.shape, divis_by=32)
        image1_pad, image2_pad = padder.pad(image1, image2)
        
        # 정밀 계산 (iters=128)
        
        _, flow_up = model(image1_pad, image2_pad, iters=raft_args.valid_iters, test_mode=True)
        
        disparity = -padder.unpad(flow_up).squeeze().cpu().numpy()
        h, w = disparity.shape
        
        points_3d = cv2.reprojectImageTo3D(disparity.astype(np.float32), Q_flipped, handleMissingValues=True)
        
        # 유효성 필터링
        z_valid = (points_3d[..., 2] > 0) & (points_3d[..., 2] < 5000)
        valid_mask = (disparity > 0.1) & np.isfinite(disparity)
        total_mask = valid_mask & z_valid
        
        if mask_img is not None:
            mask_resized = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
            total_mask = total_mask & (mask_resized > 0)
            
        valid_points = points_3d[total_mask]
        
        centroid = None
        if len(valid_points) > 0:
            centroid = np.mean(valid_points, axis=0)
            
    end = time.time()
    return centroid, end - start, len(valid_points)

# Google Cloud Vision API
API_KEY = 'AIzaSyBtTjPi2jBhiuhXNoIt_Ay3FPx_rQRMvrU'
VISION_ENDPOINT = f'https://vision.googleapis.com/v1/images:annotate?key={API_KEY}'

def analyze_object_localization(image_path):
    try:
        with open(image_path, 'rb') as img_file:
            content = base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")
        return None

    payload = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "OBJECT_LOCALIZATION", "maxResults": 10}]
        }]
    }
    
    try:
        start_time = time.time()
        response = requests.post(VISION_ENDPOINT, json=payload)
        elapsed = time.time() - start_time
        print(f"⏱️ Vision API 처리 시간: {elapsed:.2f}초")

        if response.ok:
            return response.json()
        else:
            print(f"❌ Vision API 오류: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ API 요청 중 예외 발생: {e}")
        return None

genai.configure(api_key="AIzaSyD1mxECQtGPV8XyP-Ypbq2y0D7sXe8dPbk")

def match_object_with_gemini(voice_command, detected_objects):
    objects_text = "\n".join([
        f"{i+1}. {obj['name']} (신뢰도: {obj.get('score', 0):.2f})"
        for i, obj in enumerate(detected_objects)
    ])
    prompt = (
        "다음은 이미지에서 감지된 객체 목록입니다:\n\n"
        f"{objects_text}\n\n"
        f"사용자의 음성 명령: \"{voice_command}\"\n\n"
        "목록에 해당 객체가 없으면 '없음'이라고 대답하고, "
        "있으면 가장 일치하는 객체의 번호만 숫자로 대답하세요."
    )

    try:
        start_time = time.time()
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        elapsed = time.time() - start_time
        print(f"⏱️ Gemini 처리 시간: {elapsed:.2f}초")

        text = response.text.strip()
        if '없음' in text or '없다고' in text:
            print("❌ Gemini: 해당 객체가 목록에 없습니다.")
            return None

        import re
        nums = re.findall(r'\d+', text)
        if nums:
            idx = int(nums[0]) - 1
            if 0 <= idx < len(detected_objects):
                return detected_objects[idx]
        print("❌ Gemini 응답을 이해하지 못했습니다.")
        return None
    except Exception as e:
        print(f"❌ Gemini API 오류: {e}")
        return None

# 음성 인식 시스템 클래스
class VoiceRecognitionSystem:
    def __init__(self, device_index=None):
        print("🚀 음성 인식 시스템 초기화 중...")
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.p = pyaudio.PyAudio()
        self.device_index = device_index
        self.recognizer = sr.Recognizer()
        self.rate = self.get_supported_sample_rate()
        print("🎯 시스템 준비 완료!")

    def get_supported_sample_rate(self):
        sample_rates = [44100, 48000, 22050, 16000, 8000]
        for rate in sample_rates:
            try:
                stream = self.p.open(format=self.format, channels=self.channels, rate=rate,
                                   input=True, frames_per_buffer=self.chunk, input_device_index=self.device_index)
                stream.close()
                return rate
            except: continue
        return 44100

    def print_device_list(self):
        print("\n사용 가능한 오디오 입력 장치 리스트:")
        input_devices = []
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_devices.append((i, info))
                print(f"{i}: {info['name']}")
        return input_devices

    def select_device_interactive(self):
        devices = self.print_device_list()
        if not devices: return None
        while True:
            try:
                choice = input(f"\n사용할 입력 장치 번호를 선택하세요: ")
                return int(choice)
            except: return None

    def record_audio_blocking(self):
        try:
            stream = self.p.open(format=self.format, channels=self.channels, rate=self.rate,
                                 input=True, frames_per_buffer=self.chunk, input_device_index=self.device_index)
        except Exception as e:
            print(f"❌ 오디오 장치 오류: {e}")
            return None, 0
            
        frames = []
        print("🎤 Enter 키를 눌러 녹음 시작...")
        input()
        print("🔴 녹음 시작! (종료하려면 다시 Enter)")
        
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                break
            data = stream.read(self.chunk, exception_on_overflow=False)
            frames.append(data)
            
        stream.stop_stream()
        stream.close()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            filename = tmp_file.name
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()
        return filename, 0

    def transcribe(self, wav_path, record_duration):
        if not wav_path: return None
        print("🔄 음성 인식 중...")
        try:
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio_data, language='ko-KR')
            print(f"🗣️ 인식 결과: '{text}'")
            return text
        except sr.UnknownValueError:
            print("❓ 음성을 이해하지 못했습니다.")
        except sr.RequestError as e:
            print(f"❌ 구글 음성 API 오류: {e}")
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
        return None

    def run(self):
        global voice_command, running
        try:
            while True:
                wav_path, duration = self.record_audio_blocking()
                if wav_path:
                    result = self.transcribe(wav_path, duration)
                    try: os.unlink(wav_path)
                    except: pass
                    
                    if result:
                        voice_command = result.lower()
                        print(f"🔊 명령어: {voice_command}")
                        parse_and_trigger(voice_command)
        except: pass
        finally: self.p.terminate()

def parse_and_trigger(cmd):
    global process_flag, analyze_flag, current_voice_command, tracking_active
    if '분석' in cmd or 'analyze' in cmd:
        analyze_flag = True
        current_voice_command = cmd
        print("✅ 분석 모드 활성화")
    elif '저장' in cmd or 'save' in cmd:
        process_flag = True
    elif '시작' in cmd or 'start' in cmd or '잡아' in cmd: # <--- [추가] 음성 명령 예시
        tracking_active = True
        print("🚀 [명령 수신] 로봇팔 추적/이동 시작!")

# 글로벌 변수
point = None
point_selected = False
process_flag = False
analyze_flag = False
tracking_active = False
auto_process_after_selection = False
frames_after_selection = 0
current_voice_command = ""
if_init = False      # <--- [필수 추가] 초기화 변수
rect_left = None

DISPLAY_SIZE = (720, 540)
with torch.autocast(device_type="cuda", dtype=torch.float16):
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

voice_command = ''
running = True
voice_system = VoiceRecognitionSystem(device_index=None)
device_index = 0  # [수정됨] 0번으로 고정

voice_system.device_index = device_index
threading.Thread(target=voice_system.run, daemon=True).start()

def mouse_callback(event, x, y, flags, param):
    global point, point_selected, if_init, rect_left
    h, w = rect_left.shape[:2]
    actual_x = int(x * w / DISPLAY_SIZE[0])
    actual_y = int(y * h / DISPLAY_SIZE[1])
    
    if event == cv2.EVENT_LBUTTONDOWN:
        point = [actual_x, actual_y]
        point_selected = True
        print(f"📍 선택됨: ({actual_x}, {actual_y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        point = None
        point_selected = False
        if_init = False
        print("🔄 리셋")

print("\n=== [시스템 준비 완료] ===")
print("좌클릭: 객체선택 | 's'키: 좌표계산 | 음성: '분석해줘', '저장해'")

# ====== 메인 루프 ======
frame_count = 0 # [최적화] 프레임 카운터 추가
prev_time = time.time()
while True:
    frame_count += 1
    
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    if not ret_left or not ret_right: break

    # 1. Rectification
    rect_left = cv2.remap(frame_left, left_mapx, left_mapy, cv2.INTER_LINEAR)
    rect_right = cv2.remap(frame_right, right_mapx, right_mapy, cv2.INTER_LINEAR)
    
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    h = min(h1, h2)
    w = min(w1, w2)
    rect_left = rect_left[y1:y1+h, x1:x1+w]
    rect_right = rect_right[y2:y2+h, x2:x2+w]

    try:
        # 시각화용 깊이 맵은 매 프레임 그리기 (viz_iters=10이라 빠름)
        depth_viz = get_realtime_disparity(rect_left, rect_right, viz_iters=10)
        cv2.imshow('Real-time Depth (Preview)', cv2.resize(depth_viz, DISPLAY_SIZE))
    except Exception as e:
        print(f"Depth Viz Error: {e}")

# 2. 음성 분석 로직 (API)
    if analyze_flag:
        analyze_flag = False
        timestamp = datetime.now().strftime("%H%M%S")
        temp_path = f"{save_dir}/analyze_{timestamp}.png"
        cv2.imwrite(temp_path, rect_left) 
        
        print(f"🔍 Vision API 분석 요청...")
        result = analyze_object_localization(temp_path)
        
        found_obj = False
        if result:
            responses = result.get("responses", [])
            if responses:
                objects = responses[0].get("localizedObjectAnnotations", [])
                
                # ==========================================
                # [추가됨] 감지된 객체 리스트 출력 디버깅
                # ==========================================
                print(f"\n📋 [Vision API 감지 결과 리스트] 총 {len(objects)}개 감지됨")
                if len(objects) == 0:
                    print("   👉 감지된 객체가 없습니다.")
                else:
                    for i, obj in enumerate(objects):
                        name = obj.get('name', 'Unknown')
                        score = obj.get('score', 0.0)
                        # 바운딩 박스 중심점도 같이 찍어보면 좋습니다 (선택사항)
                        vertices = obj.get("boundingPoly", {}).get("normalizedVertices", [])
                        cx_info = f"cx:{sum([v.get('x', 0) for v in vertices])/len(vertices):.2f}" if vertices else "N/A"
                        
                        print(f"   🔹 {i+1}. {name} (신뢰도: {score:.2f}, {cx_info})")
                print("-" * 40 + "\n")
                # ==========================================

                if objects:
                    best_obj = match_object_with_gemini(current_voice_command, objects)
                    if best_obj:
                        vertices = best_obj.get("boundingPoly", {}).get("normalizedVertices", [])
                        if vertices:
                            cx = sum([v.get('x', 0) for v in vertices]) / len(vertices)
                            cy = sum([v.get('y', 0) for v in vertices]) / len(vertices)
                            point = [int(cx * rect_left.shape[1]), int(cy * rect_left.shape[0])]
                            point_selected = True
                            auto_process_after_selection = True
                            frames_after_selection = 0
                            print(f"✅ 자동 선택됨: {best_obj.get('name')}")
                            found_obj = True
        
        if not found_obj:
            print("⚠️ 적절한 대상을 찾지 못했습니다.")

    # 3. SAM2 및 화면 표시
    show_frame = rect_left.copy()
    mask_img = None
    
    if point_selected:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if not globals().get('if_init', False):
                if_init = True
                predictor.load_first_frame(rect_left)
                labels = np.array([1], dtype=np.int32)
                points_arr = np.array([point], dtype=np.float32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    frame_idx=0, obj_id=(1,), points=points_arr, labels=labels
                )
            else:
                out_obj_ids, out_mask_logits = predictor.track(rect_left)
            
            out_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            mask_img = (out_mask[:, :, 0] * 255).astype(np.uint8)
            
            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            show_frame = rect_left.copy()
            cv2.drawContours(show_frame, contours, -1, (0, 165, 255), 3)  # BGR: 주황색, 두께 3
            
            if auto_process_after_selection:
                frames_after_selection += 1
                if frames_after_selection >= 10:
                    print("🎯 안정화 완료. 좌표 추적을 시작합니다.")
                    tracking_active = True
                    auto_process_after_selection = False

    cv2.imshow('Left Camera (Control)', cv2.resize(show_frame, DISPLAY_SIZE))
    cv2.setMouseCallback('Left Camera (Control)', mouse_callback)

    key = cv2.waitKey(1)
    
    # -------------------------------------------------------------------------
    # [수정됨] 연속 좌표 추적 및 전송 로직 (성능 최적화 적용)
    # -------------------------------------------------------------------------
# -------------------------------------------------------------------------
    # [수정됨] 연속 좌표 추적 및 전송 로직 (트리거 추가)
    # -------------------------------------------------------------------------
    if point_selected and mask_img is not None:
        
        # 's' 키를 눌러 추적이 활성화되었는지 확인
        if tracking_active:
            if frame_count % 3 == 0: # 3프레임마다 전송 (부하 조절)
                
                # 좌표 계산
                centroid, elapsed, n_pts = calculate_centroid_optimized(rect_left, rect_right, mask_img)
                
                if centroid is not None:
                    cx, cy, cz = centroid
                    
                    # 데이터 패키징
                    data_dict = {
                        "x": round(float(cx), 2), 
                        "y": round(float(cy), 2), 
                        "z": round(float(cz), 2)
                    }
                    json_data = json.dumps(data_dict)

                    try:
                        # 소켓 전송 (매번 열고 닫아 안전성 확보)
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        sock.sendto(json_data.encode(), ("127.0.0.1", 5005))
                        sock.close()
                        
                        # [성공 로그] 초록색 텍스트
                        print(f"\033[92m📡 [전송 성공] {json_data} (Z={cz:.1f}mm)\033[0m")
                        
                    except Exception as e:
                        # [에러 로그] 빨간색 텍스트 (이제 원인을 알 수 있음!)
                        print(f"\033[91m❌ [소켓 에러] {e}\033[0m")
                else:
                    # [계산 실패] 노란색 텍스트
                    print(f"\033[93m⚠️ [Depth 실패] 거리 측정 불가 (유효 포인트 없음)\033[0m")
        else:
            # 선택은 됐는데 's'를 안 누른 경우 (60프레임마다 알림)
            if frame_count % 60 == 0:
                print("⏸️ [대기] 객체 선택됨. 전송하려면 's' 키를 누르세요.")

    # -------------------------------------------------------------------------
    # 키 입력 처리 (중복 제거 및 정리)
    # -------------------------------------------------------------------------
    if key == ord('s'):
        tracking_active = True
        print("\n🚀 [Start] 데이터 전송을 시작합니다!")

    if key == 27: # ESC
        break
    elif key == ord('r'): # 리셋
        point_selected = False
        if_init = False
        tracking_active = False
        mask_img = None
        print("🔄 [Reset] 추적 초기화")

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()