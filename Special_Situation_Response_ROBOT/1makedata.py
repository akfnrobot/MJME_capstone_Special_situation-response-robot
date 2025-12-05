import cv2
import os
import glob

# ====== 사용자 설정 ======
left_camera_id = 2  # /dev/video2
right_camera_id = 4  # /dev/video0
TARGET_RES = (640, 480)
TARGET_FPS = 30
# ========================

# 출력 폴더 설정
output_folder_left = "./left_images"
output_folder_right = "./right_images"
os.makedirs(output_folder_left, exist_ok=True)
os.makedirs(output_folder_right, exist_ok=True)

# === 기존 이미지 삭제 ===
def clear_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, '*'))
    for f in files:
        try:
            os.remove(f)
        except IsADirectoryError:
            pass  # 폴더일 경우 무시

clear_folder(output_folder_left)
clear_folder(output_folder_right)
print('left_images, right_images 폴더 내 파일 삭제 완료')

# 좌/우 카메라 초기화 (V4L2 백엔드 강제 지정)
cap_left = cv2.VideoCapture(left_camera_id, cv2.CAP_V4L2)
cap_right = cv2.VideoCapture(right_camera_id, cv2.CAP_V4L2)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("카메라를 열 수 없습니다. 연결 상태를 확인하세요.")
    exit()

# ▼▼▼ 카메라 파라미터 설정 ▼▼▼
cap_left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap_right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_RES[0])
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_RES[1])
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_RES[0])
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_RES[1])
cap_left.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap_right.set(cv2.CAP_PROP_FPS, TARGET_FPS)

print("\n=== 카메라 설정 확인 ===")
print(f"왼쪽 해상도: {cap_left.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"오른쪽 해상도: {cap_right.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"왼쪽 FPS: {cap_left.get(cv2.CAP_PROP_FPS):.2f}")
print(f"오른쪽 FPS: {cap_right.get(cv2.CAP_PROP_FPS):.2f}")
print("=======================\n")

image_count = 0
while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("이미지를 읽을 수 없습니다.")
        break

    display_left = cv2.resize(frame_left, (840, 540))
    display_right = cv2.resize(frame_right, (840, 540))
    cv2.imshow("Left Camera (Half Size)", display_left)
    cv2.imshow("Right Camera (Half Size)", display_right)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        left_image_path = os.path.join(output_folder_left, f"left_{image_count:03d}.png")
        right_image_path = os.path.join(output_folder_right, f"right_{image_count:03d}.png")
        cv2.imwrite(left_image_path, frame_left)
        cv2.imwrite(right_image_path, frame_right)
        print(f"이미지 저장 완료: {left_image_path}, {right_image_path}")
        image_count += 1

    elif key == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
