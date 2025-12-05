from dynamixel_sdk import PortHandler

class Communication:
    def __init__(self, device_name, baudrate):
        self.portHandler = PortHandler(device_name)
        self.baudrate = baudrate
        self.is_open = False
        print("✅ 통신 핸들러가 생성되었습니다.")

    def connect(self):
        """포트를 열고 통신 속도를 설정하여 통신을 시작합니다."""
        if not self.portHandler.openPort():
            raise IOError(f"포트 열기 실패: {self.portHandler.getPortName()}")
        if not self.portHandler.setBaudRate(self.baudrate):
            raise IOError(f"통신 속도 설정 실패: {self.baudrate}")
        self.is_open = True
        print(f"'{self.portHandler.getPortName()}'에 성공적으로 연결되었습니다.")
        return True
    def disconnect(self):
        """포트를 닫습니다."""
        if self.is_open:
            self.portHandler.closePort()
            self.is_open = False
            print("포트가 닫혔습니다.")

    def get_port_handler(self):
        """외부에서 PortHandler 객체를 사용할 수 있도록 반환합니다."""
        return self.portHandler
