
import cv2
import numpy as np
import glob

# 체스보드 패턴 크기 (내부 코너 개수: 가로 9 x 세로 6)
CHECKERBOARD = (8, 6)

# 3D 체스보드 점 생성 (z=0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 3D 점과 2D 점을 저장할 리스트
objpoints = []  # 체스보드의 실제 좌표 (3D)
imgpoints_left = []  # 왼쪽 카메라의 체스보드 코너 (2D)
imgpoints_right = []  # 오른쪽 카메라의 체스보드 코너 (2D)

# 좌/우 이미지 폴더 경로 설정
left_images_path = "/home/j/stereo/left_images/*.png"
right_images_path = "/home/j/stereo/right_images/*.png"

# 좌/우 이미지 파일 리스트 가져오기
left_images = sorted(glob.glob(left_images_path))
right_images = sorted(glob.glob(right_images_path))

if len(left_images) == 0 or len(right_images) == 0:
    print("이미지가 존재하지 않습니다. 경로를 확인하세요.")
    exit()

for idx, (left_img_path, right_img_path) in enumerate(zip(left_images, right_images)):
    # 좌/우 이미지 로드
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)

    if left_img is None or right_img is None:
        print(f"이미지를 로드할 수 없습니다.\nLeft Image Path: {left_img_path}\nRight Image Path: {right_img_path}")
        continue

    # 그레이스케일 변환
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 감지
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

    if ret_left and ret_right:
        objpoints.append(objp)

        # 코너 위치를 정밀하게 조정
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_left_subpix = cv2.cornerSubPix(
            gray_left,
            corners_left,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria,
        )
        corners_right_subpix = cv2.cornerSubPix(
            gray_right,
            corners_right,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria,
        )

        imgpoints_left.append(corners_left_subpix)
        imgpoints_right.append(corners_right_subpix)

        # 실시간으로 체스보드 코너를 시각화
        cv2.drawChessboardCorners(left_img, CHECKERBOARD, corners_left_subpix, ret_left)
        cv2.drawChessboardCorners(right_img, CHECKERBOARD, corners_right_subpix, ret_right)

        # 화면에 표시
        cv2.imshow("Left Chessboard", left_img)
        cv2.imshow("Right Chessboard", right_img)

        # 저장 경로 설정 및 저장 (선택 사항)
        left_output_path = f"/home/j/stereo/visualization1/left_{idx:03d}.png"
        right_output_path = f"/home/j/stereo/visualization1/right_{idx:03d}.png"
        cv2.imwrite(left_output_path, left_img)
        cv2.imwrite(right_output_path, right_img)

        print(f"체스보드 감지 결과 저장 완료:\n{left_output_path}\n{right_output_path}")

    else:
        print(f"체스보드를 감지하지 못했습니다:\nLeft Image: {left_img_path}\nRight Image: {right_img_path}")

    # 키 입력 대기 (500ms 동안 표시 후 다음 이미지로 진행)
    if cv2.waitKey(100) & 0xFF == ord('q'):  # 'q'를 누르면 종료
        break

cv2.destroyAllWindows()
print("체스보드 감지를 완료했습니다.")

# 단일 카메라 캘리브레이션 수행 함수 정의
def calibrate_single_camera(objpoints, imgpoints, image_shape):
    """
    단일 카메라 캘리브레이션을 수행하는 함수입니다.

    Parameters:
        objpoints (list): 3D 체스보드 점 리스트
        imgpoints (list): 2D 이미지 코너 점 리스트
        image_shape (tuple): 이미지 크기 (height, width)

    Returns:
        ret (float): RMS reprojection error
        mtx (ndarray): 카메라 행렬
        dist (ndarray): 왜곡 계수
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape[::-1], None, None
    )
    return ret, mtx, dist

# 이미지 크기 설정 (이전 단계에서 처리한 이미지 크기)
image_shape = gray_left.shape

# 왼쪽 카메라 캘리브레이션 수행
ret_left, mtx_left, dist_left = calibrate_single_camera(objpoints, imgpoints_left, image_shape)

# 오른쪽 카메라 캘리브레이션 수행
ret_right, mtx_right, dist_right = calibrate_single_camera(objpoints, imgpoints_right, image_shape)

print("Left Camera Calibration Results:")
print("RMS Error:", ret_left)
print("Camera Matrix:", mtx_left)
print("Distortion Coefficients:", dist_left)

print("\nRight Camera Calibration Results:")
print("RMS Error:", ret_right)
print("Camera Matrix:", mtx_right)
print("Distortion Coefficients:", dist_right)

# 결과 데이터를 저장하는 함수 정의
def save_calibration_data(filename):
    """
    단일 카메라 캘리브레이션 데이터를 파일에 저장합니다.

    Parameters:
        filename (str): 저장할 파일 경로.
    """
    np.savez(filename,
             mtx_left=mtx_left,
             dist_left=dist_left,
             mtx_right=mtx_right,
             dist_right=dist_right)

# 결과 데이터 저장 경로 설정 및 저장
output_file_path = "/home/j/stereo/mono_calibration_data_test1.npz"
save_calibration_data(output_file_path)
print(f"Calibration data saved to {output_file_path}")
