import serial
import time
import threading
import subprocess
import os
from collections import deque
from pynput import keyboard

# Global Constants
BT_MAC = "00:22:08:31:0E:02"
BT_PORT = 1
BT_DEVICE = "/dev/rfcomm0"
BAUD_RATE = 115200

MATERIAL_IDENTIFICATION_MATRIX = {
    "Fireworks": {
        "description": "Gunpowder/Fireworks explosion",
        "CO2_range": [500, 2000],
        "Ethanol_range": [10, 100],
        "CO_range": [100, 500],
        "primary_indicator": "CO2 spike + CO spike",
        "confidence_weight": {"CO2": 0.4, "Ethanol": 0.2, "CO": 0.4}
    },
    "Gasoline Fire": {
        "description": "Gasoline combustion",
        "CO2_range": [800, 3000],
        "Ethanol_range": [500, 2000],
        "CO_range": [50, 200],
        "primary_indicator": "Ethanol rapid spike + CO2 spike",
        "confidence_weight": {"CO2": 0.3, "Ethanol": 0.5, "CO": 0.2}
    },
    "Alcohol Vapor": {
        "description": "Ethanol/Methanol leak",
        "CO2_range": [400, 600],
        "Ethanol_range": [100, 500],
        "CO_range": [10, 50],
        "primary_indicator": "High Ethanol",
        "confidence_weight": {"CO2": 0.2, "Ethanol": 0.6, "CO": 0.2}
    },
    "Incomplete Combustion": {
        "description": "Incomplete combustion / Asphyxiation",
        "CO2_range": [400, 800],
        "Ethanol_range": [20, 100],
        "CO_range": [200, 800],
        "primary_indicator": "Very High CO",
        "confidence_weight": {"CO2": 0.25, "Ethanol": 0.15, "CO": 0.6}
    },
    "Solvent/Paint Vapor": {
        "description": "Volatile Organic Solvent (VOC) leak",
        "CO2_range": [400, 700],
        "Ethanol_range": [50, 200],
        "CO_range": [20, 100],
        "primary_indicator": "Medium Ethanol + Slight CO2 rise",
        "confidence_weight": {"CO2": 0.3, "Ethanol": 0.5, "CO": 0.2}
    },
    "Normal Air": {
        "description": "No Danger",
        "CO2_range": [350, 450],
        "Ethanol_range": [0, 10],
        "CO_range": [0, 20],
        "primary_indicator": "All values near baseline",
        "confidence_weight": {"CO2": 0.33, "Ethanol": 0.33, "CO": 0.34}
    }
}

class RobotCore:
    def __init__(self):
        self.running = True
        self.window_focused = False
        self.pressed_keys = set()
        self.start_time = time.time()
        self.ser = None
        self.speed_smoothing_buffer = deque(maxlen=5)

        self.CO2_WARNING = 800.0; self.CO2_DANGER = 2000.0
        self.ETHANOL_WARNING = 50.0; self.ETHANOL_DANGER = 1000.0
        self.CO_WARNING = 15.0; self.CO_DANGER = 200.0

        self.BUFFER_SIZE = 300
        self.times = deque(maxlen=self.BUFFER_SIZE)
        self.speed_data = deque(maxlen=self.BUFFER_SIZE)
        self.co2_data = deque(maxlen=self.BUFFER_SIZE)
        self.ethanol_data = deque(maxlen=self.BUFFER_SIZE)
        self.co_data = deque(maxlen=self.BUFFER_SIZE)
        self.temp_ds_data = deque(maxlen=self.BUFFER_SIZE)
        self.temp_dht_data = deque(maxlen=self.BUFFER_SIZE)
        self.humidity_data = deque(maxlen=self.BUFFER_SIZE)
        self.identified_material_data = deque(maxlen=self.BUFFER_SIZE)

        self.current_speed = 0.0
        self.current_co2 = 0.0
        self.current_ethanol = 0.0
        self.current_co = 0.0
        self.current_temp_ds = -127.0
        self.current_temp_dht = -127.0
        self.current_humidity = -1.0
        self.current_identified_material = "Initializing..."
        self.current_material_confidence = 0.0
        self.current_power = 0

    def identify_material(self, co2_ppm, ethanol_ppm, co_ppm):
        best_match = None
        best_confidence = 0.0
        
        for material_name, specs in MATERIAL_IDENTIFICATION_MATRIX.items():
            co2_in_range = specs["CO2_range"][0] <= co2_ppm <= specs["CO2_range"][1]
            ethanol_in_range = specs["Ethanol_range"][0] <= ethanol_ppm <= specs["Ethanol_range"][1]
            co_in_range = specs["CO_range"][0] <= co_ppm <= specs["CO_range"][1]
            
            co2_c = 1.0 if co2_in_range else max(0, 1.0 - abs(co2_ppm - (specs["CO2_range"][0] if co2_ppm < specs["CO2_range"][0] else specs["CO2_range"][1])) / 1000.0)
            eth_c = 1.0 if ethanol_in_range else max(0, 1.0 - abs(ethanol_ppm - (specs["Ethanol_range"][0] if ethanol_ppm < specs["Ethanol_range"][0] else specs["Ethanol_range"][1])) / 500.0)
            co_c = 1.0 if co_in_range else max(0, 1.0 - abs(co_ppm - (specs["CO_range"][0] if co_ppm < specs["CO_range"][0] else specs["CO_range"][1])) / 300.0)
            
            weights = specs["confidence_weight"]
            total_confidence = (co2_c * weights["CO2"] + eth_c * weights["Ethanol"] + co_c * weights["CO"])
            
            if total_confidence > best_confidence:
                best_confidence = total_confidence
                best_match = material_name
        
        if best_confidence < 0.5: best_match = "Unidentified"
        return best_match, best_confidence

    def auto_bind(self):
        if os.path.exists(BT_DEVICE): return
        try:
            subprocess.run(["sudo", "rfcomm", "bind", BT_DEVICE, BT_MAC, str(BT_PORT)], capture_output=True, timeout=5)
        except: pass

    def connect_serial(self):
        print(f"[*] Connecting to Tank: {BT_DEVICE}")
        try:
            self.ser = serial.Serial(BT_DEVICE, BAUD_RATE, timeout=1)
            time.sleep(2)
            return True
        except: return False

    def parse_sensor_line(self, line):
        try:
            parts = line.split('|')
            for part in parts:
                part = part.strip()
                if part.startswith("S:"):
                    val = float(part.replace('S:', '').replace('m/s', '').strip())
                    self.speed_smoothing_buffer.append(val)
                    self.current_speed = sum(self.speed_smoothing_buffer)/len(self.speed_smoothing_buffer)
                elif part.startswith("CO2:"): self.current_co2 = float(part.replace('CO2:', '').replace('ppm', ''))
                elif part.startswith("E:"): self.current_ethanol = float(part.replace('E:', '').replace('ppm', ''))
                elif part.startswith("CO:"): self.current_co = float(part.replace('CO:', '').replace('ppm', ''))
                elif part.startswith("T:"): self.current_temp_ds = float(part.replace('T:', '').replace('C', ''))
                elif part.startswith("T2:"): self.current_temp_dht = float(part.replace('T2:', '').replace('C', ''))
                elif part.startswith("H:"): self.current_humidity = float(part.replace('H:', '').replace('%', ''))
            
            self.current_identified_material, self.current_material_confidence = self.identify_material(
                self.current_co2, self.current_ethanol, self.current_co
            )
            
            self.times.append(time.time() - self.start_time)
            self.speed_data.append(self.current_speed)
            self.co2_data.append(self.current_co2)
            self.ethanol_data.append(self.current_ethanol)
            self.co_data.append(self.current_co)
            self.temp_ds_data.append(self.current_temp_ds)
            self.temp_dht_data.append(self.current_temp_dht)
            self.humidity_data.append(self.current_humidity)
            self.identified_material_data.append(self.current_identified_material)
        except: pass

    def read_sensor_data_thread(self):
        buffer = ""
        while self.running:
            try:
                if self.ser and self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.startswith('S:') and '|' in line:
                            self.parse_sensor_line(line)
            except: pass
            time.sleep(0.02)

    def send_command(self, command):
        if self.ser:
            try: self.ser.write(command.encode())
            except: pass

    def on_press(self, key):
        if not self.window_focused: return
        try:
            if key == keyboard.Key.esc: return False
            if key in self.pressed_keys: return
            self.pressed_keys.add(key)
            
            if hasattr(key, 'char'):
                c = key.char
                # 탱크 이동 (WASD)
                if c == 'w': self.send_command('w')
                elif c == 's': self.send_command('s')
                elif c == 'a': self.send_command('a')
                elif c == 'd': self.send_command('d')
                
                # 그리퍼 (G/H)
                elif c == 'g': self.send_command('G')
                elif c == 'h': self.send_command('F')
                
                # 속도 (0-9)
                elif c in '0123456789':
                    self.current_power = int(c)
                    self.send_command(c)
        except: pass

    def on_release(self, key):
        try:
            if key == keyboard.Key.esc:
                self.send_command('x')
                self.running = False
                return False
            if not self.window_focused: return
            self.pressed_keys.discard(key)
            
            if hasattr(key, 'char') and key.char in 'wasd':
                self.send_command('x')
        except: pass