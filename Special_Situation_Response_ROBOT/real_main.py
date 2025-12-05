import sys
import time
import numpy as np
import os
import socket   # UDP 통신용
import json     # 데이터 파싱용
import select   # Non-blocking 소켓 감지용

# [1] arm_module의 ArmController를 가져와서 하드웨어 제어를 위임합니다.
from arm_module import ArmController 

# --------------------------------------------------------------------------
# [플랫폼별 키 입력 함수]
# --------------------------------------------------------------------------
if os.name == 'nt':  # Windows
    import msvcrt
    def get_key():
        if msvcrt.kbhit():
            return msvcrt.getch().decode().lower()
        return None
else:  # Linux / Mac
    import sys, tty, termios, select
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

# --------------------------------------------------------------------------
# [설정] 좌표계 정의: X=Left(+), Y=Up(+), Z=Forward(+)
# --------------------------------------------------------------------------
EXTRA_CONFIG = {
    "CAM_OFFSET_vec": np.array([25.00, 25.00, 225.00]), 
    "CAMERA_TILT_DEG": 10.0,
    "HOME_Z_DIST": 275.5, 
    "UDP_PORT": 5005,
    "GRIPPER_PORT": 5007,
    "SAG_COMPENSATION_Y": 10.0,   # 중력 처짐 보정값 (mm) 
     
    # [설정 수정] 거리 및 속도 관련 파라미터
    "FAST_ZONE_LIMIT": 300.0,     # 30cm(300mm) 이상일 때 FAST 모드
    "VISION_CUTOFF_GAP": 80.0,    # 사거리 도달 5cm 전부터는 Vision 무시하고 진입
    
    "MOVE_FAST_MM": 55.0,        # 스피드 5, 0.2초 이동 거리
    "MOVE_SLOW_MM": 7.0          # 스피드 4, 0.1초 이동 거리
}

# --------------------------------------------------------------------------
# [Vision Helper Functions]
# --------------------------------------------------------------------------
def get_vision_coordinates_from_file(filepath="vision_data.txt"):
    try:
        with open(filepath, "r") as f:
            line = f.read().strip()
            if not line: return None
            data = list(map(float, line.split()))
            return data 
    except:
        return None

def vision_to_wrist_coords(raw_data):
    # 1. Meter -> mm 변환
    cam_x = raw_data[0] * 1000
    cam_y = raw_data[1] * 1000
    cam_z = raw_data[2] * 1000

    # 2. 틸트(10도) 보정
    rad = np.deg2rad(EXTRA_CONFIG["CAMERA_TILT_DEG"])
    sin_t = np.sin(rad)
    cos_t = np.cos(rad)

    cam_y_world_down = cam_y * cos_t + cam_z * sin_t
    cam_z_world_fwd  = -cam_y * sin_t + cam_z * cos_t
    cam_x_world_right = cam_x 

    # 3. 좌표계 매핑
    raw_x_calc = EXTRA_CONFIG["CAM_OFFSET_vec"][0] - cam_x_world_right
    robot_x = -1.0 * raw_x_calc
    robot_y = EXTRA_CONFIG["CAM_OFFSET_vec"][1] - cam_y_world_down
    robot_z = EXTRA_CONFIG["CAM_OFFSET_vec"][2] + cam_z_world_fwd

    return np.array([robot_x, robot_y, robot_z])

# --------------------------------------------------------------------------
# [EOD Mission Logic]
# --------------------------------------------------------------------------
class EOD_Mission_Control:
    
    TANK_CONTROL_IP = "127.0.0.1"
    TANK_CONTROL_PORT = 5007
    TARGET_DISTANCE_LIMIT = 300
    
    def __init__(self, arm_controller):
        self.arm_ctrl = arm_controller
        self.mc = arm_controller.controller
        self.kin = arm_controller.kinematics
        self.m_ids = arm_controller.motor_ids
        self.home_pos = arm_controller.home_positions
        self.vels = arm_controller.vels
        self.ENABLE_PHYSICAL_STRIKE = False 
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_tank_command(self, command, duration_ms=0):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(command.encode(), (self.TANK_CONTROL_IP, self.TANK_CONTROL_PORT))
            print(f"🚦 [Auto Drive] Command '{command}' sent.")
            if duration_ms > 0:
                time.sleep(duration_ms / 1000.0)
                sock.sendto('x'.encode(), (self.TANK_CONTROL_IP, self.TANK_CONTROL_PORT)) 
                print(f"🚦 [Auto Drive] Command 'x' (STOP) sent after {duration_ms}ms.")
        except Exception as e:
            print(f"❌ Tank Command Send Error: {e}")
        finally:
            sock.close()

    def apply_smart_correction(self, angle_deg):
        threshold = 45.0 
        if abs(angle_deg) <= threshold:
            return angle_deg
        else:
            sign = 1 if angle_deg > 0 else -1
            base = threshold * sign
            excess = angle_deg - base
            return base + (excess * 0.5)

    def send_gripper_command(self, cmd_char):
        try:
            self.udp_sock.sendto(cmd_char.encode(), ("127.0.0.1", EXTRA_CONFIG["GRIPPER_PORT"]))
            print(f"📨 [Gripper] 명령 전송: '{cmd_char}' -> TankControl")
        except Exception as e:
            print(f"❌ 그리퍼 전송 실패: {e}")

    def execute_strike(self, target_x, target_y, target_z):
        target_y += EXTRA_CONFIG["SAG_COMPENSATION_Y"]
        if not self.ENABLE_PHYSICAL_STRIKE:
            print("\n🔒 [Safe Mode] 이동 비활성화.")
            print(f"   -> 목표 좌표: Left(X)={target_x:.1f}, Up(Y)={target_y:.1f}, Fwd(Z)={target_z:.1f}")
            return False

        print(f"\n🚀 [Move] 원본 목표: X={target_x:.1f}, Y={target_y:.1f}, Z={target_z:.1f}")
        dist_origin = np.sqrt(target_x**2 + target_y**2 + target_z**2)
        MAX_REACH_LIMIT = 269.5 

        if dist_origin > MAX_REACH_LIMIT:
            scale_factor = MAX_REACH_LIMIT / dist_origin
            target_x *= scale_factor
            target_y *= scale_factor
            target_z *= scale_factor
            print(f"🔥 [Force Reach] 거리 조정: {dist_origin:.1f}mm -> {MAX_REACH_LIMIT}mm")

        res = self.kin.solve_ik(target_x, target_y, target_z)
        if res:
            th_x, th_y = res
            corr_x = np.deg2rad(self.apply_smart_correction(np.rad2deg(th_x)))
            corr_y = np.deg2rad(self.apply_smart_correction(np.rad2deg(th_y)))
            
            goals = self.kin.calculate_pulse_goals(corr_x, corr_y, self.home_pos)
            goals_int = {mid: goals[f'goal_p{i+1}'] for i, mid in enumerate(self.m_ids)}
            
            self.mc.move_motors_sync(self.m_ids, goals_int, self.vels, 800)
            self.arm_ctrl.curr_tx = np.rad2deg(corr_x)
            self.arm_ctrl.curr_ty = np.rad2deg(corr_y)
            self.arm_ctrl.last_tx, self.arm_ctrl.last_ty = self.arm_ctrl.curr_tx, self.arm_ctrl.curr_ty
            
            print("⏳ 이동 중...")
            time.sleep(2.0) 
            print("✊ [Auto Grip] 물체 잡기 시도...")
            self.send_gripper_command('g')
            time.sleep(1.0)
            print("✅ 동작 완료.")
            return True
        else:
            print("⚠️ [IK Fatal Error] 해를 찾을 수 없습니다.")
            return False

def flush_udp(sock):
    while True:
        r, _, _ = select.select([sock], [], [], 0)
        if r:
            try: sock.recvfrom(4096)
            except: break
        else: break

# =================================================================================
# [Main Execution]
# =================================================================================
def main():
    print("\n🚀 [EOD Robot Control System] Initializing...")
    

    arm = ArmController()
    if not arm.connect():
        print("❌ Arm 모듈 연결 실패.")
        return

    origin_positions = {mid: arm.home_positions[f'home_p{i+1}'] for i, mid in enumerate(arm.motor_ids)}

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        udp_sock.bind(("0.0.0.0", EXTRA_CONFIG["UDP_PORT"]))
        udp_sock.setblocking(False)
        print(f"📡 [UDP Server] 좌표 수신 대기 (Port: {EXTRA_CONFIG['UDP_PORT']})...")
    except Exception as e:
        print(f"❌ UDP 설정 실패: {e}")
        return

    eod_mission = EOD_Mission_Control(arm)
    eod_mission.ENABLE_PHYSICAL_STRIKE = False

    print("\n---------------------------------------------------")
    print(" 📡  [AUTO]    : capture3에서 'S' 누르면 자동 잡기")
    print(" 🔄  [R]       : 리셋 (원점 복귀)")
    print(" 🕹️   [I/J/K/L/M]: 수동 제어")
    print(" 🤏  [G/F]     : 그리퍼 잡기/풀기")
    print(" ❌  [Q]       : 종료")
    print("---------------------------------------------------")

    last_target_x = 0.0
    last_target_y = 0.0
    last_target_z = None
    is_holding_object = False 

    try:
        while True:
            try:
                ready = select.select([udp_sock], [], [], 3.0)
                
                if is_holding_object:
                    if ready[0]: flush_udp(udp_sock)
                    pass 
                
                # =========================================================
                # [상황 A] Vision Mode
                # =========================================================
                elif ready[0]:
                    data, _ = udp_sock.recvfrom(1024)
                    if data:
                        coords = json.loads(data.decode())
                        raw_vec = [coords['x'], coords['y'], coords['z']]
                        wrist_coords = vision_to_wrist_coords(raw_vec)
                        
                        target_x, target_y, target_z = wrist_coords
                        last_target_x, last_target_y, last_target_z = target_x, target_y, target_z

                        R = (target_x**2 + target_z**2)**0.5 
                        print(f"👀 [Vision] Dist: {R:.1f}mm | Tgt: Z={target_z:.1f}")

                        # ------------------------------------------------------------------
                        # [로직 1] 30cm 기준 속도 분기 (FAST_ZONE_LIMIT)
                        # ------------------------------------------------------------------
                        if R > EXTRA_CONFIG["FAST_ZONE_LIMIT"]:
                            print(f"🚀 [FAST ZONE] 거리 {R:.1f}mm > {EXTRA_CONFIG['FAST_ZONE_LIMIT']}mm. 고속 접근.")
                            eod_mission.send_tank_command('4')
                            eod_mission.send_tank_command('w', duration_ms=400) # 0.2s
                            last_target_z -= EXTRA_CONFIG["MOVE_FAST_MM"]
                            
                            time.sleep(0.1)
                            flush_udp(udp_sock)
                            continue

                        # ------------------------------------------------------------------
                        # [로직 2] 사정거리 내 진입 여부 판단
                        # ------------------------------------------------------------------
                        if R > eod_mission.TARGET_DISTANCE_LIMIT:
                            gap = R - eod_mission.TARGET_DISTANCE_LIMIT
                            print(f"🐢 [SLOW ZONE] 거리 {R:.1f}mm. (남은 거리: {gap:.1f}mm)")

                            # --------------------------------------------------------------
                            # [핵심 로직] 5cm(50mm) 이하 -> Vision 중단 -> 계산된 시간만큼 전진
                            # --------------------------------------------------------------
                            if gap <= EXTRA_CONFIG["VISION_CUTOFF_GAP"]:
                                print(f"\n⚠️ [FINAL APPROACH] 남은 거리 {gap:.1f}mm가 5cm 이내입니다.")
                                print("🚫 카메라 데이터 신뢰 불가 -> Vision 루프 탈출 및 예상 전진 수행.")
                                
                                # 비례식: 5mm : 100ms = gap : needed_duration
                                time_ratio = 100.0 / EXTRA_CONFIG["MOVE_SLOW_MM"]
                                needed_duration = int(gap * time_ratio)
                                
                                if needed_duration < 50: needed_duration = 50 

                                print(f"   >> 🏁 마지막 {gap:.1f}mm 전진 ({needed_duration}ms)")
                                eod_mission.send_tank_command('4')
                                eod_mission.send_tank_command('w', duration_ms=needed_duration)
                                
                                last_target_z -= gap
                                target_z = last_target_z
                                
                                # continue 없이 아래 잡기 로직으로 바로 진입
                                
                            else:
                                # 5cm보다 멀면 일반 Slow 전진
                                print(f"   >> 정밀 접근 중... ({EXTRA_CONFIG['MOVE_SLOW_MM']}mm 전진)")
                                eod_mission.send_tank_command('4')
                                eod_mission.send_tank_command('w', duration_ms=100)
                                last_target_z -= EXTRA_CONFIG["MOVE_SLOW_MM"]
                                
                                time.sleep(0.1)
                                flush_udp(udp_sock)
                                continue 
                        
                        # ==========================================================
                        # [잡기 실행]
                        # ==========================================================
                        print("✅ [Action] 잡기 시퀀스 시작.")
                        eod_mission.send_tank_command('x')
                        time.sleep(0.5)
                        
                        temp_safe = eod_mission.ENABLE_PHYSICAL_STRIKE
                        eod_mission.ENABLE_PHYSICAL_STRIKE = True 
                        
                        strike_success = eod_mission.execute_strike(target_x, target_y, target_z)
                        
                        eod_mission.ENABLE_PHYSICAL_STRIKE = temp_safe
                        
                        if strike_success:
                            print("🚗 [Auto Reverse] 물체 확보 완료. 2초간 후진...")
                            eod_mission.send_tank_command('s')
                            time.sleep(2.0) 
                            eod_mission.send_tank_command('x') 
                            is_holding_object = True
                            print("🏁 종료.\n")
                        else:
                            print("❌ 실패. 재시도 대기.")
                        
                        time.sleep(2.0)
                        flush_udp(udp_sock)
                        last_target_z = None 

                # =========================================================
                # [상황 B] Blind Mode (Vision Lost)
                # =========================================================
                else:
                    if is_holding_object: pass
                    elif last_target_z is None: pass
                    else:
                        R = (last_target_x**2 + last_target_z**2)**0.5
                        print(f"\n👻 [Blind] 시각 정보 소실! 추정 거리: {R:.1f}mm")

                        if R > eod_mission.TARGET_DISTANCE_LIMIT:
                            gap = R - eod_mission.TARGET_DISTANCE_LIMIT
                            
                            # [버그 수정됨] 여기서 continue가 없어서 잡기가 실행됐던 문제 해결
                            if R > EXTRA_CONFIG["FAST_ZONE_LIMIT"]:
                                print(f"   >> 🚀 [Blind/FAST] {EXTRA_CONFIG['MOVE_FAST_MM']}mm 예상 전진")
                                eod_mission.send_tank_command('5')
                                eod_mission.send_tank_command('w', duration_ms=200)
                                last_target_z -= EXTRA_CONFIG["MOVE_FAST_MM"]
                                
                                time.sleep(0.1)
                                continue # <--- [중요] 이 코드가 없어서 바로 아래 잡기로 넘어갔었음!
                            else:
                                if gap <= EXTRA_CONFIG["VISION_CUTOFF_GAP"]:
                                    print("   >> 🏁 [Blind] 목표 지점 도달 간주. 전진 후 잡기.")
                                    time_ratio = 100.0 / EXTRA_CONFIG["MOVE_SLOW_MM"]
                                    needed_duration = int(gap * time_ratio)
                                    if needed_duration < 50: needed_duration = 50

                                    eod_mission.send_tank_command('4')
                                    eod_mission.send_tank_command('w', duration_ms=needed_duration)
                                    last_target_z -= gap
                                else:
                                    print(f"   >> 🐢 [Blind/SLOW] {EXTRA_CONFIG['MOVE_SLOW_MM']}mm 예상 전진")
                                    eod_mission.send_tank_command('4')
                                    eod_mission.send_tank_command('w', duration_ms=100)
                                    last_target_z -= EXTRA_CONFIG["MOVE_SLOW_MM"]
                                    time.sleep(0.5)
                                    continue
                        
                        # [Blind 잡기 실행]
                        print("✅ [Blind Action] 맹목적 잡기 시도.")
                        eod_mission.send_tank_command('x')
                        time.sleep(0.5)
                        temp_safe = eod_mission.ENABLE_PHYSICAL_STRIKE
                        eod_mission.ENABLE_PHYSICAL_STRIKE = True 
                        
                        strike_success = eod_mission.execute_strike(last_target_x, last_target_y, last_target_z)
                        
                        eod_mission.ENABLE_PHYSICAL_STRIKE = temp_safe

                        if strike_success:
                            print("🚗 [Auto Reverse] 성공. 후진.")
                            eod_mission.send_tank_command('s')
                            time.sleep(2.0)
                            eod_mission.send_tank_command('x')
                            is_holding_object = True
                            print("🏁 Blind 종료.\n")
                        else:
                            print("❌ Blind 실패.")

                        time.sleep(2.0)
                        flush_udp(udp_sock)
                        last_target_z = None

            except json.JSONDecodeError: pass

            key = get_key()
            if key:
                if key in ['i', 'j', 'k', 'l', 'm']:
                    arm.set_key_state(key, True)
                    time.sleep(0.05) 
                    arm.set_key_state(key, False)
                elif key == 'r':
                    print("\n🔄 [Reset]")
                    arm.controller.move_motors_sync(arm.motor_ids, origin_positions, arm.vels, 800)
                    arm.curr_tx, arm.curr_ty = 0.0, 0.0

                    # 상태 플래그 / 마지막 좌표 완전 초기화
                    is_holding_object = False
                    last_target_x = 0.0
                    last_target_y = 0.0
                    last_target_z = None   # <-- 이게 중요: None이면 Vision/Bliind 둘 다 '아직 타겟 없음'으로 인식

                    print("🔓 잡기 상태 해제 + 타겟 좌표 초기화. 새 무게중심 대기.")
                    time.sleep(1.0)

                elif key == 'g': eod_mission.send_gripper_command('g')
                elif key == 'f': 
                    eod_mission.send_gripper_command('h')
                    if is_holding_object: is_holding_object = False
                elif key == 'q': break
            time.sleep(0.01)

    except KeyboardInterrupt: print("\n강제 종료.")
    finally:
        arm.close()
        udp_sock.close()

if __name__ == "__main__":
    main()