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
