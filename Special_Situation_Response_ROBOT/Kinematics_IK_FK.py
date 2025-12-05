import numpy as np

class RobotKinematics:
    def __init__(self, robot_params):
        self.params = robot_params
        
        # 그리퍼 길이 (없으면 0)
        self.L_grip = robot_params.get('L_grip', 0.0)
        
        # 모터 펄스 변환 상수
        self.MM_PER_DEGREE = 0.1547
        self.PULSES_PER_DEGREE = 4095.0 / 360.0
        
        print(f"✅ 로봇 기구학 로드 완료 (L_grip={self.L_grip}mm)")

    def calculate_pulse_goals(self, theta_x_rad, theta_y_rad, home_positions):
        # 1. 길이 변화량 계산 (Right, Left, Top, Bottom 순서 반환됨)
        d_right, d_left, d_top, d_bottom = self._calculate_4_tendon_lengths(theta_x_rad, theta_y_rad)

        # 2. 펄스 변환 및 방향 적용
        # ID 1 (Left)
        p_left_pulse = int(d_left / self.MM_PER_DEGREE * self.PULSES_PER_DEGREE) * -1
        
        # ID 3 (Right)
        p_right_pulse = int(d_right / self.MM_PER_DEGREE * self.PULSES_PER_DEGREE) * 1
        
        # ID 4 (Top)
        p_top_pulse = int(d_top / self.MM_PER_DEGREE * self.PULSES_PER_DEGREE) * 1
        
        # ID 2 (Bottom)
        p_bottom_pulse = int(d_bottom / self.MM_PER_DEGREE * self.PULSES_PER_DEGREE) * 1

        # 3. 목표 딕셔너리 생성
        # 메인 코드의 motor_ids 순서: [Left(1), Right(3), Top(4), Bottom(2)]
        # 이에 맞춰 goal_p1 ~ goal_p4 할당
        goals = {
            'goal_p1': home_positions['home_p1'] + p_left_pulse,   # ID 1
            'goal_p2': home_positions['home_p2'] + p_right_pulse,  # ID 3
            'goal_p3': home_positions['home_p3'] + p_top_pulse,    # ID 4
            'goal_p4': home_positions['home_p4'] + p_bottom_pulse  # ID 2
        }
        return goals

    def _calculate_4_tendon_lengths(self, theta_x, theta_y):
        """
        기하학적 길이 변화량 계산
        Returns: d_right, d_left, d_top, d_bottom
        """
        d = self.params['d']
        theta_d = np.sqrt(theta_x**2 + theta_y**2)
        
        if theta_d < 1e-6: return 0, 0, 0, 0
            
        phi = np.arctan2(theta_y, -theta_x)
        
        # 0:Right, pi:Left, pi/2:Top, 3pi/2:Bottom
        deltas = [0, np.pi, np.pi/2, 3*np.pi/2]
        
        elongations = []
        for delta in deltas:
            # -d * theta * cos(...) : 음수면 길이가 줄어듦(당김)
            val = -d * theta_d * np.cos(delta - phi)
            elongations.append(val)
            
        return elongations[0], elongations[1], elongations[2], elongations[3]

    def calculate_fk(self, theta_x, theta_y):
        """ 현재 각도 -> [x, y, z] 좌표 반환 """
        n = self.params['n']
        l = self.params['l']
        dp3 = self.params.get('dp3', 0)
        
        theta_d = np.sqrt(theta_x**2 + theta_y**2)
        phi = np.arctan2(theta_y, -theta_x)
        
        r_bb = 0.0
        z_bb = 0.0
        
        # 척추 계산
        for i in range(1, n + 1):
            theta_i = i * (theta_d / n)
            r_bb += l * np.sin(theta_i) + dp3 * np.sin(theta_i + theta_d/(2*n))
            z_bb += l * np.cos(theta_i) + dp3 * np.cos(theta_i + theta_d/(2*n))
            
        # 그리퍼 오프셋 추가
        r_target = r_bb + self.L_grip * np.sin(theta_d)
        z_target = z_bb + self.L_grip * np.cos(theta_d)
        
        # 3D 변환
        x = r_target * np.cos(phi)
        y = r_target * np.sin(phi)
        z = z_target
        
        return x, y, z

    def solve_ik(self, target_x, target_y, target_z):
        # 1. DLS 풀이
        r_target = np.sqrt(target_x**2 + target_y**2)
        target_pos_xz = np.array([r_target, target_z])
        
        L_total = (self.params['n'] * self.params['l']) + self.L_grip
        theta_guess = 2 * np.arctan2(r_target, target_z)
        if abs(theta_guess) < 1e-4: theta_guess = 0.001
        
        current_theta_d = theta_guess
        lambda_val = 0.5; alpha = 0.2
        
        for i in range(1000):
            r_curr, z_curr = self._fk_xz_plane(current_theta_d)
            error = target_pos_xz - np.array([r_curr, z_curr])
            
            if np.linalg.norm(error) < 0.005: break # 수렴
            
            J = self._jacobian_xz(current_theta_d)
            JJT = np.dot(J, J.T)
            damped_inv = np.linalg.inv(JJT + lambda_val**2 * np.eye(2))
            delta = np.dot(J.T, np.dot(damped_inv, error))
            current_theta_d += alpha * delta[0]

        # 2. 각도 변환
        phi = np.arctan2(target_y, target_x)
        theta_x_sol = -current_theta_d * np.cos(phi)
        theta_y_sol = current_theta_d * np.sin(phi)
        
        final_x, final_y, final_z = self.calculate_fk(theta_x_sol, theta_y_sol)
        
        # 목표점과의 거리(오차) 계산 - 각 축 별 개별 지정
        error_x = abs(target_x - final_x)
        error_y = abs(target_y - final_y)
        error_z = abs(target_z - final_z)
        
        # 각 축별 허용 오차 임계값 (mm)
        ERROR_THRESHOLD_X = 5  # X축 허용 오차
        ERROR_THRESHOLD_Y = 5  # Y축 허용 오차
        ERROR_THRESHOLD_Z = 25  # Z축 허용 오차
        
        # 각 축 중 하나라도 임계값 초과하면 실패
        if error_x > ERROR_THRESHOLD_X or error_y > ERROR_THRESHOLD_Y or error_z > ERROR_THRESHOLD_Z:
            print(f"   ⚠️ [IK 기각] 오차 초과 -> X: {error_x:.2f}mm (허용: {ERROR_THRESHOLD_X}), Y: {error_y:.2f}mm (허용: {ERROR_THRESHOLD_Y}), Z: {error_z:.2f}mm (허용: {ERROR_THRESHOLD_Z}) -> 이동 취소")
            return None  # 실패 반환 -> 모터 안 움직임

        return theta_x_sol, theta_y_sol

    def _fk_xz_plane(self, theta_d):
        n = self.params['n']
        l = self.params['l']
        dp3 = self.params.get('dp3', 0)
        
        r = 0.0; z = 0.0
        for i in range(1, n+1):
            th_i = i * (theta_d / n)
            r += l * np.sin(th_i) + dp3 * np.sin(th_i + theta_d/(2*n))
            z += l * np.cos(th_i) + dp3 * np.cos(th_i + theta_d/(2*n))
            
        # 그리퍼 반영
        r += self.L_grip * np.sin(theta_d)
        z += self.L_grip * np.cos(theta_d)
        return r, z
        
    def _jacobian_xz(self, theta_d):
        h = 1e-6
        r1, z1 = self._fk_xz_plane(theta_d)
        r2, z2 = self._fk_xz_plane(theta_d + h)
        
        dr = (r2 - r1) / h
        dz = (z2 - z1) / h
        
        # J는 2x1 행렬 (입력: theta_d 1개, 출력: r, z 2개)
        return np.array([[dr], [dz]])