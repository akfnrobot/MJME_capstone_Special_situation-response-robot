import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pynput import keyboard
import threading
import sys
import time
import socket
# 모듈 Import (사용자 환경에 맞게 유지)
from robotcore import RobotCore, MATERIAL_IDENTIFICATION_MATRIX
from arm_module import ArmController 


# --- [통합 수정] 주행 + 그리퍼 + 속도 제어 리스너 ---
def unified_udp_listener(robot_core):
    udp_ip = "127.0.0.1"
    udp_port = 5007
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((udp_ip, udp_port))
        print(f"👂 [Unified Listener] 포트 {udp_port}에서 모든 명령 대기 중...")
        
        while True:
            data, _ = sock.recvfrom(1024)
            cmd = data.decode().strip()
            
            if cmd in ['w', 's', 'a', 'd', 'x']:
                robot_core.send_command(cmd)
            elif cmd.upper() == 'G':
                robot_core.send_command('G')
            elif cmd.upper() == 'H':
                robot_core.send_command('F')
            elif cmd in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                robot_core.current_power = int(cmd)
                robot_core.send_command(cmd)
            else:
                print(f"⚠️ 알 수 없는 명령: {cmd}")


    except Exception as e:
        print(f"❌ UDP Listener Error: {e}")


def main():
    # 1. 로봇 Core 및 Arm 초기화
    robot = RobotCore()
    arm = ArmController()
    
    # 2. 연결 시도
    robot.auto_bind()
    if not robot.connect_serial():
        print("[Warning] Tank Serial connect failed. Running in Offline Mode.")


    if not arm.connect():
        print("[Warning] Robot Arm connect failed.")


    # 3. 데이터 수신 스레드
    sensor_thread = threading.Thread(target=robot.read_sensor_data_thread, daemon=True)
    sensor_thread.start()


    t_unified = threading.Thread(target=unified_udp_listener, args=(robot,), daemon=True)
    t_unified.start()
    
    # 4. 그래프 초기화
    fig, axs = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle('Real-time Tank Sensor Monitoring - EOD/HAZMAT Detection System',
                fontsize=20, fontweight='bold')


    # ---------------------------------------------------------
    # [수정 1] 이벤트 핸들러: 포커스 해제 시 로봇팔도 멈추도록 수정
    # ---------------------------------------------------------
    def on_focus(event): 
        robot.window_focused = True
        
    def on_blur(event): 
        robot.window_focused = False
        robot.pressed_keys.clear()
        robot.send_command('x') # 탱크 정지
        robot.current_power = 0
        
        # [추가됨] 창 밖으로 나가면 로봇팔 동작 강제 종료 (안전장치)
        for key_char in ['i', 'j', 'k', 'l', 'm']:
            arm.set_key_state(key_char, False)
    
    fig.canvas.mpl_connect('figure_enter_event', on_focus)
    fig.canvas.mpl_connect('figure_leave_event', on_blur)


    # 5. 애니메이션 함수 (기존 로직 유지)
    def animate_wrapper(i):
        if len(robot.times) == 0: return


        # 폰트 설정
        value_font_size = 20
        legend_font_size = 14
        ylabel_font_size = 22
        xlabel_font_size = 16
        tick_font_size = 16
        text_box_font_size = 15
        
        # ===== 1. Speed Graph =====
        axs[0, 0].clear()
        axs[0, 0].plot(robot.times, robot.speed_data, 'b', label='Velocity (m/s)', linewidth=2)
        axs[0, 0].set_ylim(0, 5)
        if robot.speed_data:
            axs[0, 0].text(0.98, 0.95, f'{robot.current_speed:.2f} m/s\n({robot.current_speed*3.6:.1f} km/h)',
                          transform=axs[0, 0].transAxes, fontsize=value_font_size, fontweight='bold',
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axs[0, 0].legend(loc='upper left', fontsize=legend_font_size)
        axs[0, 0].set_ylabel('m/s', fontsize=ylabel_font_size, fontweight='bold')
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].tick_params(labelsize=tick_font_size)


        # ===== 2. CO2 Graph =====
        axs[0, 1].clear()
        axs[0, 1].plot(robot.times, robot.co2_data, 'g', label='CO2 (ppm)', linewidth=2)
        axs[0, 1].axhline(y=robot.CO2_WARNING, color='orange', linestyle='--', linewidth=2.5, label=f'Warning: {robot.CO2_WARNING}ppm')
        axs[0, 1].axhline(y=robot.CO2_DANGER, color='red', linestyle='--', linewidth=2.5, label=f'Danger: {robot.CO2_DANGER}ppm')
        
        if robot.co2_data:
            if robot.current_co2 > robot.CO2_DANGER:
                co2_status, box_color, text_color = "DANGER", 'red', 'white'
            elif robot.current_co2 > robot.CO2_WARNING:
                co2_status, box_color, text_color = "WARNING", 'yellow', 'black'
            else:
                co2_status, box_color, text_color = "NORMAL", 'lightgreen', 'black'
            axs[0, 1].text(0.98, 0.95, f'{robot.current_co2:.1f} ppm\n{co2_status}',
                          transform=axs[0, 1].transAxes, fontsize=value_font_size, fontweight='bold',
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8), color=text_color)
        axs[0, 1].legend(loc='upper left', fontsize=legend_font_size)
        axs[0, 1].set_ylabel('ppm', fontsize=ylabel_font_size, fontweight='bold')
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].tick_params(labelsize=tick_font_size)


        # ===== 3. Ethanol Graph =====
        axs[1, 0].clear()
        axs[1, 0].plot(robot.times, robot.ethanol_data, 'r', label='Ethanol (ppm)', linewidth=2)
        axs[1, 0].axhline(y=robot.ETHANOL_WARNING, color='orange', linestyle='--', linewidth=2.5, label=f'Warning: {robot.ETHANOL_WARNING}ppm')
        axs[1, 0].axhline(y=robot.ETHANOL_DANGER, color='red', linestyle='--', linewidth=2.5, label=f'Danger: {robot.ETHANOL_DANGER}ppm')
        
        if robot.ethanol_data:
            if robot.current_ethanol > robot.ETHANOL_DANGER:
                ethanol_status, box_color, text_color = "DANGER", 'red', 'white'
            elif robot.current_ethanol > robot.ETHANOL_WARNING:
                ethanol_status, box_color, text_color = "WARNING", 'yellow', 'black'
            else:
                ethanol_status, box_color, text_color = "NORMAL", 'lightblue', 'black'
            axs[1, 0].text(0.98, 0.95, f'{robot.current_ethanol:.1f} ppm\n{ethanol_status}',
                          transform=axs[1, 0].transAxes, fontsize=value_font_size, fontweight='bold',
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8), color=text_color)
        axs[1, 0].legend(loc='upper left', fontsize=legend_font_size)
        axs[1, 0].set_ylabel('ppm', fontsize=ylabel_font_size, fontweight='bold')
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].tick_params(labelsize=tick_font_size)


        # ===== 4. CO Graph =====
        axs[1, 1].clear()
        axs[1, 1].plot(robot.times, robot.co_data, 'purple', label='CO (ppm)', linewidth=2)
        axs[1, 1].axhline(y=robot.CO_WARNING, color='orange', linestyle='--', linewidth=2.5, label=f'Warning: {robot.CO_WARNING}ppm')
        axs[1, 1].axhline(y=robot.CO_DANGER, color='red', linestyle='--', linewidth=2.5, label=f'Danger: {robot.CO_DANGER}ppm')


        if robot.co_data:
            if robot.current_co > robot.CO_DANGER:
                co_status, box_color, text_color = "DANGER", 'red', 'white'
            elif robot.current_co > robot.CO_WARNING:
                co_status, box_color, text_color = "WARNING", 'yellow', 'black'
            else:
                co_status, box_color, text_color = "NORMAL", 'lightyellow', 'black'
            axs[1, 1].text(0.98, 0.95, f'{robot.current_co:.2f} ppm\n{co_status}',
                          transform=axs[1, 1].transAxes, fontsize=value_font_size, fontweight='bold',
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8), color=text_color)
        axs[1, 1].legend(loc='upper left', fontsize=legend_font_size)
        axs[1, 1].set_ylabel('ppm', fontsize=ylabel_font_size, fontweight='bold')
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].tick_params(labelsize=tick_font_size)


        # ===== 5. Temperature Graph =====
        axs[2, 0].clear()
        axs[2, 0].plot(robot.times, robot.temp_ds_data, 'b', label='DS18B20 (Base)', linewidth=2)
        axs[2, 0].plot(robot.times, robot.temp_dht_data, 'orange', label='DHT22 (Ambient)', linewidth=2)
        axs[2, 0].set_ylim(-30, 60)
        
        if robot.temp_ds_data and robot.temp_dht_data:
            text_str = f'DS18B20: {robot.current_temp_ds:.2f}C\nDHT22: {robot.current_temp_dht:.2f}C'
            axs[2, 0].text(0.98, 0.95, text_str,
                          transform=axs[2, 0].transAxes, fontsize=value_font_size-2, fontweight='bold',
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axs[2, 0].legend(loc='upper left', fontsize=legend_font_size)
        axs[2, 0].set_ylabel('Celsius', fontsize=ylabel_font_size, fontweight='bold')
        axs[2, 0].grid(True, alpha=0.3)
        axs[2, 0].tick_params(labelsize=tick_font_size)


        # ===== 6. Humidity & Material ID =====
        axs[2, 1].clear()
        axs[2, 1].axis('off')
        
        ax_humidity = axs[2, 1].inset_axes([0, 0, 0.48, 1.0])
        ax_text = axs[2, 1].inset_axes([0.52, 0, 0.48, 1.0])
        
        ax_humidity.plot(robot.times, robot.humidity_data, 'c', label='Humidity (%)', linewidth=2)
        ax_humidity.set_ylim(0, 100)
        if robot.humidity_data:
            ax_humidity.text(0.95, 0.95, f'{robot.current_humidity:.1f} %',
                          transform=ax_humidity.transAxes, fontsize=value_font_size-2, fontweight='bold',
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.8))
        ax_humidity.legend(loc='upper left', fontsize=legend_font_size)
        ax_humidity.set_ylabel('%', fontsize=ylabel_font_size, fontweight='bold')
        ax_humidity.grid(True, alpha=0.3)
        ax_humidity.tick_params(labelsize=tick_font_size)
        ax_humidity.set_xlabel('Time (s)', fontsize=xlabel_font_size, fontweight='bold')
        
        ax_text.clear()
        ax_text.axis('off')
        
        if robot.current_material_confidence > 0.7:
            bg_color = 'lightcyan'
            confidence_level = "High"
        elif robot.current_material_confidence > 0.4:
            bg_color = 'lightyellow'
            confidence_level = "Medium"
        else:
            bg_color = 'lightcoral'
            confidence_level = "Low"
        
        identification_text = (
            f"Material Identification\n"
            f"{'='*20}\n"
            f"Identified:\n{robot.current_identified_material}\n\n"
            f"Confidence: {robot.current_material_confidence*100:.1f}%\n"
            f"Conf. Level: {confidence_level}\n"
            f"Robot Power: {robot.current_power} / 9"
        )
        
        ax_text.text(0.5, 0.5, identification_text,
                       transform=ax_text.transAxes,
                       fontsize=text_box_font_size,
                       fontweight='bold',
                       verticalalignment='center',
                       horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.9, pad=1),
                       family='monospace')
        
        for ax in axs[0:2, :].flat:
            ax.set_xlabel('Time (s)', fontsize=xlabel_font_size, fontweight='bold')
        axs[2, 0].set_xlabel('Time (s)', fontsize=xlabel_font_size, fontweight='bold')
        
        plt.tight_layout()
    print(" [Arm] I(Up)/M(Down)/J(Left)/L(Right)/K(Reset)")
    print(" Click graph window to enable controls.")


    plt.rcParams['keymap.save'] = ''
    ani = FuncAnimation(fig, animate_wrapper, interval=200)
    
    # -----------------------------------------------------
    # [수정 2] 통합 키 핸들러: 포커스 여부 체크 추가
    # -----------------------------------------------------
    def integrated_on_press(key):
        # 모니터링 창이 선택(Focus)되지 않았으면 키 입력 무시
        if not robot.window_focused:
            return


        robot.on_press(key) # 탱크 처리
        
        # 로봇팔 (I, J, K, L, M) 처리
        try:
            if hasattr(key, 'char') and key.char:
                c = key.char.lower()
                if c in ['i', 'j', 'k', 'l', 'm']:
                    arm.set_key_state(c, True)
        except: pass


    def integrated_on_release(key):
        # 떼는 동작은 포커스와 상관없이 처리하거나, 포커스일 때만 처리
        # 여기서는 press와 동일하게 포커스일 때만 작동하도록 함
        # (on_blur에서 이미 안전장치를 마련했기 때문)
        if not robot.window_focused:
            return
            
        robot.on_release(key) # 탱크 처리
        
        try:
            if hasattr(key, 'char') and key.char:
                c = key.char.lower()
                if c in ['i', 'j', 'k', 'l', 'm']:
                    arm.set_key_state(c, False)
        except: pass


    listener = keyboard.Listener(on_press=integrated_on_press, on_release=integrated_on_release)
    listener.start()
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n🔄 [Shutdown] Ctrl+C 감지. 로봇팔 원점 복귀 중...")
    finally:
        print("\nShutting down...")
        
        # [추가됨] 종료 직전에 로봇팔 원점 복귀
        try:
            print("🔙 [Arm Home] 로봇팔을 원점으로 이동 중...")
            origin_positions = {mid: arm.home_positions[f'home_p{i+1}'] for i, mid in enumerate(arm.motor_ids)}
            arm.controller.move_motors_sync(arm.motor_ids, origin_positions, arm.vels, 800)
            time.sleep(2.0)  # 원점 복귀 완료 대기
            print("✅ 로봇팔 원점 복귀 완료.")
        except Exception as e:
            print(f"⚠️ 로봇팔 원점 복귀 중 오류: {e}")
        
        robot.running = False
        arm.close()
        listener.stop()
        if robot.ser:
            robot.send_command('x')
            time.sleep(0.2)
            robot.ser.close()
        
        print("👋 프로그램 종료.")


if __name__ == "__main__":
    main()