#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import os

# ------------------------------------------------------------------
# [핵심 수정] OS별 키보드 입력 처리 클래스 (Context Manager)
# ------------------------------------------------------------------
if os.name == 'nt':  # Windows
    import msvcrt
    class Console:
        def __enter__(self): return self
        def __exit__(self, type, value, traceback): pass
        def get_key(self):
            if msvcrt.kbhit():
                return msvcrt.getch().decode().lower()
            return None
else:  # Linux / Mac
    import sys, tty, termios, select
    class Console:
        def __init__(self):
            self.old_settings = None

        def __enter__(self):
            # 프로그램 시작 시 설정을 저장하고 Raw 모드로 변경
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            return self

        def __exit__(self, type, value, traceback):
            # 프로그램 종료 시 설정 복구
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

        def get_key(self):
            # 입력 확인 (Non-blocking)
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1).lower()
            return None

from communication import Communication as CM
from control import MotorController as MC

CONFIG = {
    # [포트 설정]
    "DEVICENAME": '/dev/ttyUSB0',
    "BAUDRATE": 57600,
    "PWM_LIMIT": 500,
    
    "MOTORS": {
        # 물리적 위치 (1이 좌측, 3이 우측)
        1: "Left   (좌 - ID 1)",
        3: "Right  (우 - ID 3)",
        2: "Bottom (하 - ID 2)",
        4: "Top    (상 - ID 4)"
    },
    
    # W가 감기도록 방향 부호 반전됨
    "WINDING_DIR": { 1: 1, 2: -1, 3: -1, 4: -1 }
}

def print_info(mid, step):
    # 한 줄 덮어쓰기를 위해 줄바꿈 대신 포맷팅 사용
    sys.stdout.write(f"\r👉 [ID {mid}] {CONFIG['MOTORS'][mid]} 선택됨 (Step: {step})          ")
    sys.stdout.flush()

def main():
    # 1. 통신 연결 시도 (안전한 방식)
    try:
        comm = CM(CONFIG["DEVICENAME"], CONFIG["BAUDRATE"])
        # connect() 결과값 체크를 제거하고 호출만 수행 (에러 없으면 성공으로 간주)
        comm.connect()
        print(f"✅ 통신 연결 성공: {CONFIG['DEVICENAME']}")
    except Exception as e:
        print(f"❌ 초기화 오류: {e}")
        return

    controller = MC(comm)
    motor_ids = list(CONFIG["MOTORS"].keys())
    
    try:
        current_positions = {}
        
        # 2. 모터 초기화
        print("⚡ 모터 초기화 중...")
        for mid in motor_ids:
            try:
                controller.initialize_motor(mid, CONFIG["PWM_LIMIT"])
                pos = controller.read_present_position(mid)
                current_positions[mid] = pos if pos is not None else 0
            except Exception as e:
                print(f"⚠️ ID {mid} 초기화 실패: {e}")
                current_positions[mid] = 0

        # 기본값 설정
        selected_id = 1
        step_size = 20 
        
        print("\n---------------------------------------------------")
        print("⚡ [Motor Setting Tool] - 방향 최종 수정 (Linux/Win)")
        print("---------------------------------------------------")
        print("   1~4         : 모터 선택")
        print("   W (꾹 누름) : 🧵 감기 (Tighten)")
        print("   S (꾹 누름) : 🌀 풀기 (Release)")
        print("   X / Z       : 🚀 고속 / 🔍 정밀 모드")
        print("   Space       : 🛑 토크 해제")
        print("   Q           : 👋 종료")
        print("---------------------------------------------------")
        
        # 초기 상태 출력
        print_info(selected_id, step_size)

        # [핵심] Console Context Manager 사용
        with Console() as console:
            while True:
                # 키 입력 감지 (빠름)
                key = console.get_key()
                
                if key:
                    # 1. 모터 선택
                    if key in ['1', '2', '3', '4']:
                        selected_id = int(key)
                        print_info(selected_id, step_size)

                    # 2. 속도 모드
                    elif key == 'x': 
                        step_size = 50 
                        print_info(selected_id, step_size)
                        
                    elif key == 'z': 
                        step_size = 20 
                        print_info(selected_id, step_size)

                    # 3. 이동 (W=감기, S=풀기)
                    elif key == 'w' or key == 's':
                        direction = CONFIG["WINDING_DIR"][selected_id]
                        action_str = ""
                        
                        if key == 'w':
                            # 감기 (방향 적용)
                            current_positions[selected_id] += (step_size * direction)
                            action_str = "W:감기"
                        elif key == 's':
                            # 풀기
                            current_positions[selected_id] -= (step_size * direction)
                            action_str = "S:풀기"
                        
                        # 명령 전송 (동기화 없이 단일 명령 전송 가능하지만, 여기선 SyncWrite 유지)
                        goals = {mid: current_positions[mid] for mid in motor_ids}
                        vels = {mid: 500 for mid in motor_ids} 
                        controller.move_motors_sync(motor_ids, goals, vels, 0)
                        
                        # 화면 갱신 (제자리 출력)
                        # \033[K 는 커서 위치부터 줄 끝까지 지우는 터미널 코드입니다 (잔상 제거용)
                        sys.stdout.write(f"\r[{CONFIG['MOTORS'][selected_id]}] Pos: {int(current_positions[selected_id])} | {action_str} \033[K")
                        sys.stdout.flush()

                    # 4. 종료 및 토크 해제
                    elif key == ' ': # Space bar
                        for mid in motor_ids: controller.set_torque(mid, False)
                        sys.stdout.write("\n🛑 토크 해제됨\n")
                        time.sleep(0.5)
                        
                    elif key == 'q': 
                        sys.stdout.write("\n👋 종료\n")
                        break

                time.sleep(0.01)

    except Exception as e:
        print(f"\n[오류] {e}")
    finally:
        print("\n🔌 연결 종료 중...")
        try:
            comm.disconnect()
        except:
            pass

if __name__ == "__main__":
    main()