import time
import threading
import numpy as np
import sys

# 기존 라이브러리 import
try:
    from communication import Communication as CM
    from control import MotorController as MC
    from Kinematics_IK_FK import RobotKinematics as RK
except ImportError:
    pass

class ArmController:
    def __init__(self):
        self.running = True
        self.enabled = False
        
        # [keyboardd.py 원본 설정값 복원]
        self.CONFIG = {
            "DEVICENAME": '/dev/ttyUSB0', # 장치 관리자에서 포트 확인 필요 (윈도우면 COMx)
            "BAUDRATE": 57600,
            "DXL_ID_LEFT": 1,
            "DXL_ID_RIGHT": 3,
            "DXL_ID_TOP": 4,
            "DXL_ID_BOTTOM": 2,
            "robot_params": {
                'n': 6, 'l': 34.02, 'R': 17.01, 'B': 7.37, 'd': 23.67, 'dp3': 0, 
                'L_grip': 70.0 
            },
            "FIXED_VEL_VAL": 200,
            "PROFILE_ACCELERATION_MS": 100,
            "PWM_LIMITS": { 1: 400, 3: 400, 4: 400, 2: 400 },
            "STEP_DEGREE": 1.5,     # 움직임 단위
            "MAX_ANGLE": 110.0       # 최대 각도 제한
        }

        # 현재 좌표 및 목표 좌표
        self.curr_tx = 0.0 # Roll
        self.curr_ty = 0.0 # Pitch
        self.last_tx = 0.0
        self.last_ty = 0.0
        
        # 키 입력 상태 저장소 (I, J, K, L, M)
        self.key_states = {
            'i': False, # Up
            'm': False, # Down
            'j': False, # Left
            'l': False, # Right
            'k': False  # Reset
        }

        # 객체 초기화
        self.comm = None
        self.controller = None
        self.kinematics = None
        self.motor_ids = [1, 2, 3, 4]
        self.home_positions = {}
        self.vels = {}

    def connect(self):
        try:
            if 'CM' not in globals(): raise ImportError
            
            print("[Arm] Connecting to Dynamixels...")
            self.comm = CM(self.CONFIG["DEVICENAME"], self.CONFIG["BAUDRATE"])
            self.comm.connect()
            
            self.controller = MC(self.comm)
            self.kinematics = RK(self.CONFIG["robot_params"])
            
            self.motor_ids = [
                self.CONFIG["DXL_ID_LEFT"], self.CONFIG["DXL_ID_RIGHT"], 
                self.CONFIG["DXL_ID_TOP"], self.CONFIG["DXL_ID_BOTTOM"]
            ]

            # 모터 초기화 및 홈 포지션 읽기
            for mid in self.motor_ids:
                try:
                    self.controller.initialize_motor(mid, self.CONFIG["PWM_LIMITS"].get(mid, 300))
                except: pass

            for i, mid in enumerate(self.motor_ids):
                pos = self.controller.read_present_position(mid)
                self.home_positions[f'home_p{i+1}'] = pos

            self.vels = {mid: self.CONFIG["FIXED_VEL_VAL"] for mid in self.motor_ids}
            self.enabled = True
            print("[Arm] Connected & Ready.")
            
            # 독립 스레드 시작 (keyboardd.py의 while loop 역할)
            threading.Thread(target=self._control_loop, daemon=True).start()
            return True

        except Exception as e:
            print(f"[Arm] Init Failed: {e}")
            self.enabled = False
            return False

    def _control_loop(self):
        """keyboardd.py의 메인 루프를 그대로 구현"""
        step = self.CONFIG["STEP_DEGREE"]
        
        while self.running and self.enabled:
            try:
                updated = False

                # 1. 키 상태 확인 및 좌표 계산 (I, J, K, L, M)
                if self.key_states['k']: # K: Reset to Origin
                    self.curr_tx = 0.0
                    self.curr_ty = 0.0
                    updated = True
                else:
                    if self.key_states['i']: # I: Up (Pitch +)
                        self.curr_ty += step
                        updated = True
                    if self.key_states['m']: # M: Down (Pitch -)
                        self.curr_ty -= step
                        updated = True
                    if self.key_states['j']: # J: Left (Roll -)
                        self.curr_tx += step
                        updated = True
                    if self.key_states['l']: # L: Right (Roll +)
                        self.curr_tx -= step
                        updated = True

                # 2. 각도 제한 (Clipping)
                self.curr_tx = np.clip(self.curr_tx, -self.CONFIG["MAX_ANGLE"], self.CONFIG["MAX_ANGLE"])
                self.curr_ty = np.clip(self.curr_ty, -self.CONFIG["MAX_ANGLE"], self.CONFIG["MAX_ANGLE"])

                # 3. 변화가 있을 때만 모터 명령 전송 (Threshold 0.01)
                if abs(self.curr_tx - self.last_tx) > 0.01 or abs(self.curr_ty - self.last_ty) > 0.01:
                    rad_x = np.deg2rad(self.curr_tx)
                    rad_y = np.deg2rad(self.curr_ty)
                    
                    goals = self.kinematics.calculate_pulse_goals(rad_x, rad_y, self.home_positions)
                    
                    final_goals = {
                        self.CONFIG["DXL_ID_LEFT"]:   goals['goal_p1'],
                        self.CONFIG["DXL_ID_RIGHT"]:  goals['goal_p2'],
                        self.CONFIG["DXL_ID_TOP"]:    goals['goal_p3'],
                        self.CONFIG["DXL_ID_BOTTOM"]: goals['goal_p4']
                    }
                    
                    self.controller.move_motors_sync(self.motor_ids, final_goals, self.vels, self.CONFIG["PROFILE_ACCELERATION_MS"])
                    self.last_tx, self.last_ty = self.curr_tx, self.curr_ty
                
                # 원본과 동일한 루프 주기 (0.01초 = 100Hz)
                time.sleep(0.01)

            except Exception as e:
                print(f"[Arm Loop Error] {e}")
                time.sleep(1)

    def set_key_state(self, key_char, is_pressed):
        """키보드 입력을 받아 상태 갱신"""
        if key_char in self.key_states:
            self.key_states[key_char] = is_pressed

    def close(self):
        self.running = False
        if self.enabled:
            try:
                for mid in self.motor_ids: 
                    self.controller.set_torque(mid, False)
                self.comm.disconnect()
            except: pass