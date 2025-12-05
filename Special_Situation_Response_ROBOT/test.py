import serial
import time
import subprocess
import os
import sys

# -------------------------------------------------------
# [설정] robotcore.py와 동일한 블루투스 설정
# -------------------------------------------------------
BT_MAC = "00:22:08:31:0E:02"
BT_PORT = 1
BT_DEVICE = "/dev/rfcomm0"
BAUD_RATE = 115200

# 테스트 파라미터
TEST_SPEED = '4'        # 속도 설정
MOVE_DURATION = 0.1    # 전진 시간 (초)

# -------------------------------------------------------
# [유틸] 키 입력 감지 함수 (사용자가 주신 코드 기반)
# -------------------------------------------------------
if os.name == 'nt':  # Windows
    import msvcrt
    def get_key():
        if msvcrt.kbhit():
            return msvcrt.getch().decode().lower()
        return None
else:  # Linux / Mac
    import tty, termios, select
    def get_key():
        dr, dw, de = select.select([sys.stdin], [], [], 0)
        if dr:
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                return ch.lower()
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        return None

# -------------------------------------------------------
# [기능] 블루투스 및 시리얼 연결
# -------------------------------------------------------
def auto_bind():
    if os.path.exists(BT_DEVICE): return
    print(f"[*] 블루투스 바인딩 시도... ({BT_MAC})")
    try:
        subprocess.run(["sudo", "rfcomm", "bind", BT_DEVICE, BT_MAC, str(BT_PORT)], capture_output=True, timeout=5)
        time.sleep(1)
    except: pass

def connect_tank():
    auto_bind()
    try:
        print(f"[*] 탱크 연결 중: {BT_DEVICE}...")
        ser = serial.Serial(BT_DEVICE, BAUD_RATE, timeout=1)
        time.sleep(2) # 아두이노 리셋 대기
        print("[*] 연결 성공!\n")
        return ser
    except Exception as e:
        print(f"[!] 연결 실패: {e}")
        return None

# -------------------------------------------------------
# [메인] 반복 테스트 로직
# -------------------------------------------------------
def main():
    ser = connect_tank()
    if not ser: return

    print("============================================")
    print(f" 📏 탱크 이동 거리 측정 (속도: {TEST_SPEED}, 시간: {MOVE_DURATION}초)")
    print("============================================")
    print(" [T] : 테스트 시작 (전진 후 정지)")
    print(" [Q] : 프로그램 종료")
    print("--------------------------------------------")

    try:
        while True:
            key = get_key()
            
            if key == 't':
                print(f"\n🚀 [동작] {MOVE_DURATION}초간 전진합니다...")
                
                # 1. 속도 설정
                ser.write(TEST_SPEED.encode())
                time.sleep(0.05)

                # 2. 전진 시작
                start_time = time.time()
                ser.write(b'w')

                # 3. 정해진 시간만큼 대기 (Busy wait for precision)
                while (time.time() - start_time) < MOVE_DURATION:
                    pass
                
                # 4. 정지
                ser.write(b'x')
                print(f"🛑 [정지] 완료. 거리를 측정하세요.")
                print("   (다시 하려면 't'를 누르세요)")

                # 버퍼 비우기 (불필요한 데이터 제거)
                ser.reset_input_buffer()

            elif key == 'q':
                print("\n👋 프로그램을 종료합니다.")
                ser.write(b'x') # 안전을 위해 한번 더 정지
                break
            
            time.sleep(0.01) # CPU 점유율 방지

    except KeyboardInterrupt:
        print("\n[!] 강제 종료")
    finally:
        if ser: ser.close()

if __name__ == "__main__":
    main()