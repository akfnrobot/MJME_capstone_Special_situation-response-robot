import cv2
import numpy as np
import glob
import os
import time

# #############################################################
# 사용자 설정 값 (실제 환경에 맞게 수정 필수!)
# #############################################################
CHECKERBOARD = (8, 6)          # 체스보드 내부 코너 수 (가로 x 세로)
SQUARE_SIZE = 0.015          # 체스보드 사각형 한 변의 실제 크기(미터)
CALIB_DATA_PATH = '/home/j/stereo/stereo_calibration_data.npz'
LEFT_IMAGE_DIR = "/home/j/stereo/left_images/*.png"
RIGHT_IMAGE_DIR = "/home/j/stereo/right_images/*.png"
SHOW_SEC = 0.1                  # 코너 시각화 자동 넘김 시간(초)
# #############################################################

def print_matrix(name, matrix):
    """행렬 출력 포매팅 함수"""
    print(f"\n{name}:")
    np.set_printoptions(precision=6, suppress=True)
    print(matrix)

# 3D 체스보드 점 생성 (z=0, SQUARE_SIZE 반영)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

# 데이터 저장 구조 초기화
objpoints, imgpoints_left, imgpoints_right = [], [], []

# 이미지 로드 및 코너 검출 ---------------------------------------
print("\n[1/4] 체스보드 검출 시작 (자동 시각화)")
left_images = sorted(glob.glob(LEFT_IMAGE_DIR))
right_images = sorted(glob.glob(RIGHT_IMAGE_DIR))

for left_path, right_path in zip(left_images, right_images):
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        print(f"경고: {left_path} 또는 {right_path} 로드 실패")
        continue

    # 코너 검출 (좌/우 동시 성공 시만 처리)
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD)

    if ret_left and ret_right:
        # 서브픽셀 정확도 향상
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)

        # 코너 시각화 (자동 넘김)
        debug_left = cv2.drawChessboardCorners(left_img.copy(), CHECKERBOARD, corners_left, ret_left)
        debug_right = cv2.drawChessboardCorners(right_img.copy(), CHECKERBOARD, corners_right, ret_right)

        # 원하는 크기로 리사이즈 (예: 가로 960px, 세로 비율 유지)
        scale_width = 960
        scale_height_left = int(debug_left.shape[0] * scale_width / debug_left.shape[1])
        scale_height_right = int(debug_right.shape[0] * scale_width / debug_right.shape[1])

        debug_left_resized = cv2.resize(debug_left, (scale_width, scale_height_left))
        debug_right_resized = cv2.resize(debug_right, (scale_width, scale_height_right))

        cv2.imshow('Left Corners', debug_left_resized)
        cv2.imshow('Right Corners', debug_right_resized)
        cv2.waitKey(int(SHOW_SEC * 1000))  # 1초(조절 가능)
        cv2.destroyWindow('Left Corners')
        cv2.destroyWindow('Right Corners')


        objpoints.append(objp)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)
        print(f"○ 성공: {os.path.basename(left_path)} | {os.path.basename(right_path)}")
    else:
        print(f"✕ 실패: {os.path.basename(left_path)} | {os.path.basename(right_path)}")

cv2.destroyAllWindows()

if len(objpoints) < 5:
    print("체스보드 검출 성공 쌍이 너무 적습니다. 더 많은 이미지를 촬영하세요.")
    exit()

# 캘리브레이션 수행 --------------------------------------------
print("\n[2/4] 단안 캘리브레이션 (초기값 설정 추가)")
img_shape = gray_left.shape[::-1]

# 초기 카메라 매트릭스 추정값 설정
initial_cam_matrix = np.array([
    [img_shape[0], 0, img_shape[0]/2],
    [0, img_shape[1], img_shape[1]/2],
    [0, 0, 1]
], dtype=np.float32)

# 단안 캘리브레이션 (초기값 사용)
ret_left, mtx_left, dist_left, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_left, img_shape, 
    cameraMatrix=initial_cam_matrix.copy(),
    distCoeffs=None,
    flags=cv2.CALIB_USE_INTRINSIC_GUESS
)
ret_right, mtx_right, dist_right, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_right, img_shape,
    cameraMatrix=initial_cam_matrix.copy(),
    distCoeffs=None,
    flags=cv2.CALIB_USE_INTRINSIC_GUESS
)

print("\n[단안 결과]")
print(f"- 왼쪽 RMS: {ret_left:.4f} 픽셀")
print(f"- 오른쪽 RMS: {ret_right:.4f} 픽셀")
print_matrix("왼쪽 내부 파라미터", mtx_left)
print_matrix("왼쪽 왜곡 계수", dist_left)
print_matrix("오른쪽 내부 파라미터", mtx_right)
print_matrix("오른쪽 왜곡 계수", dist_right)

# 스테레오 캘리브레이션 ----------------------------------------
print("\n[3/4] 스테레오 캘리브레이션 (플래그 수정)")
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_left, dist_left, mtx_right, dist_right,
    img_shape, 
    criteria=criteria, 
    flags=cv2.CALIB_USE_INTRINSIC_GUESS  # ✅

)

print("\n[스테레오 결과]")
print(f"- RMS: {ret_stereo:.4f} 픽셀 (정상 범위: 0.1 ~ 2.0)")
print_matrix("회전 행렬 (R)", R)
print_matrix("이동 벡터 (T) [미터]", T)
print(f"베이스라인 거리: {abs(T[0][0]):.3f}m (일반적으로 0.05~0.5m)")

# 스테레오 정류화 계산 ----------------------------------------
print("\n[4/4] 스테레오 정류화 (ROI 보정)")
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_left, dist_left, 
    mtx_right, dist_right,
    img_shape, R, T, 
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0  # 넓은 시야 유지
)

print_matrix("재투영 행렬 (Q)", Q)

# 매핑 테이블 생성
mapL1, mapL2 = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, img_shape, cv2.CV_32FC1)
mapR1, mapR2 = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, img_shape, cv2.CV_32FC1)

# 데이터 저장
np.savez(CALIB_DATA_PATH,
    mtx_left=mtx_left, dist_left=dist_left,
    mtx_right=mtx_right, dist_right=dist_right,
    R=R, T=T, E=E, F=F,
    R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
    roi1=roi1, roi2=roi2
)
print(f"\n캘리브레이션 데이터 저장: {CALIB_DATA_PATH}")

# 검증 이미지 생성 --------------------------------------------
print("\n[5/5] 검증 이미지 생성 (높이 보정 추가)")
output_dirs = {
    'rectified': '/home/j/stereo/rectified_image',
    'epipolar': '/home/j/stereo/epipolar_image'
}
for d in output_dirs.values():
    os.makedirs(d, exist_ok=True)

for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    # 정류화 적용
    left_rect = cv2.remap(left_img, mapL1, mapL2, cv2.INTER_LANCZOS4)
    right_rect = cv2.remap(right_img, mapR1, mapR2, cv2.INTER_LANCZOS4)

    # ROI 크롭 (좌우 별도 처리)
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    
    left_rect = left_rect[y1:y1+h1, x1:x1+w1]
    right_rect = right_rect[y2:y2+h2, x2:x2+w2]

    # 이미지 저장 전 유효성 검사
    if left_rect.size == 0 or right_rect.size == 0:
        print(f"경고: {idx}번 이미지 크롭 실패. ROI 확인 필요")
        continue

    # 높이 불일치 보정 (추가)
    min_height = min(left_rect.shape[0], right_rect.shape[0])
    left_cropped = left_rect[:min_height, :]
    right_cropped = right_rect[:min_height, :]

    # 정류화 이미지 저장
    cv2.imwrite(f"{output_dirs['rectified']}/left_rect_{idx:03d}.png", left_cropped)
    cv2.imwrite(f"{output_dirs['rectified']}/right_rect_{idx:03d}.png", right_cropped)
    
    # 에피폴라 라인 시각화
    combined = np.hstack((left_cropped, right_cropped))
    h, w = combined.shape[:2]
    for y in range(0, h, 25):
        cv2.line(combined, (0,y), (w-1,y), (0,255,0), 1)
    
    cv2.imwrite(f"{output_dirs['epipolar']}/epipolar_{idx:03d}.png", combined)

print("모든 프로세스 완료!")
