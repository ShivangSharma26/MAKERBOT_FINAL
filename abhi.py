import cv2
import numpy as np
import time
import threading
import serial
import os
from mbcameramanager import CameraManager

# --- Face Tracker Class ---
class FaceTracker(threading.Thread):
    def __init__(self, camera_manager, ui_callback=None):
        super().__init__(daemon=True)
        self.camera_manager = camera_manager
        self.ui_callback = ui_callback
        self.running = True
        self.face_tracking_active = False
        self.display_window = True

        self.servo_center = 90
        self.servo_range = 40
        self.x_min = self.servo_center - self.servo_range
        self.x_max = self.servo_center + self.servo_range
        self.y_min = self.servo_center - self.servo_range
        self.y_max = self.servo_center + self.servo_range

        self.x_buffer = []
        self.y_buffer = []
        self.buffer_size = 5
        self.last_good_x = self.servo_center
        self.last_good_y = self.servo_center
        self.last_face_seen_time = time.time()

        # Use Raspberry Pi 5 port
        try:
            self.serial_port = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            time.sleep(2)
            self.log("✅ Connected to Arduino on /dev/ttyACM0")
        except Exception as e:
            self.serial_port = None
            self.log(f"❌ Could not open serial port: {e}")

        self.current_neck_angle = self.servo_center

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def log(self, message):
        if self.ui_callback:
            self.ui_callback(f"🎯 FACE TRACKER: {message}")
        else:
            print(f"🎯 FACE TRACKER: {message}")

    def map_range(self, val, in_min, in_max, out_min, out_max):
        return int(np.interp(val, [in_min, in_max], [out_min, out_max]))

    def smooth_angle(self, buffer, new_value):
        buffer.append(new_value)
        if len(buffer) > self.buffer_size:
            buffer.pop(0)
        return int(np.mean(buffer))

    def send_servo_command(self, neck_target, eye_angle, y_angle):
        if not self.serial_port or not self.serial_port.is_open:
            self.log_to_ui("⚠️ Serial port not open.")
            return

        neck_target = max(min(neck_target, self.servo_center + 55), self.servo_center - 55)
        eye_angle = max(min(eye_angle, self.servo_center + 20), self.servo_center - 20)
        y_angle = max(min(y_angle, self.y_max), self.y_min)

        step = 7         # Increased step size
        delay = 0.0005   # Reduced delay for faster transitions

        if abs(neck_target - self.current_neck_angle) > 1:
            direction = 1 if neck_target > self.current_neck_angle else -1
            for angle in range(self.current_neck_angle, neck_target, direction * step):
                command = f"N:{angle} E:{eye_angle} Y:{y_angle}\n"
                try:
                    self.serial_port.write(command.encode())
                    time.sleep(delay)
                except Exception as e:
                    self.log(f"⚠️ Serial write error: {e}")
                    break
            self.current_neck_angle = neck_target

        else:
            command = f"N:{neck_target} E:{eye_angle} Y:{y_angle}\n"
            try:
                self.serial_port.write(command.encode())
            except Exception as e:
                self.log(f"⚠️ Serial write error: {e}")

    def process_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
         largest_face = max(faces, key=lambda r: r[2] * r[3])
         x, y, w, h = largest_face
         center_x = x + w // 2
         center_y = y + h // 2
         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
         return False


        x_angle = self.smooth_angle(self.x_buffer, self.map_range(center_x, 0, frame.shape[1], self.x_min, self.x_max))
        y_angle = self.smooth_angle(self.y_buffer, self.map_range(center_y, 0, frame.shape[0], self.y_max, self.y_min))

        self.last_good_x = x_angle
        self.last_good_y = y_angle
        self.last_face_seen_time = time.time()

        x_norm = (x_angle - self.servo_center) / ((self.x_max - self.x_min) / 2)
        eye_angle = int(self.servo_center + (x_norm * 20))

        edge_threshold = 0.35
        left_edge = frame.shape[1] * edge_threshold
        right_edge = frame.shape[1] * (1 - edge_threshold)

        if center_x < left_edge:
            self.current_neck_angle = max(self.current_neck_angle - 2, self.servo_center - 55)
        elif center_x > right_edge:
            self.current_neck_angle = min(self.current_neck_angle + 2, self.servo_center + 55)

        self.send_servo_command(self.current_neck_angle, eye_angle, y_angle)

        if not self.face_tracking_active:
            self.log("Face tracking STARTED")
            self.face_tracking_active = True

        return True

    def run(self):
        self.log("Initializing face tracker...")
        while self.running:
            frame = self.camera_manager.get_frame()
            if frame is None:
                continue

            frame = cv2.flip(frame, 1)
            face_found = self.process_faces(frame)
            
           


            if not face_found:
                if time.time() - self.last_face_seen_time < 5.0:
                    self.send_servo_command(
                        self.current_neck_angle,
                        self.servo_center,
                        self.last_good_y
                    )
                elif self.face_tracking_active:
                    self.log("Face not detected for 5 seconds — tracking stopped.")
                    self.face_tracking_active = False

    def stop(self):
        self.running = False
        self.log("Face tracker stopped")
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        try:
            cv2.destroyAllWindows()
        except:
            pass

# --- Main Entry Point ---
if __name__ == "__main__":
    try:
        cam = CameraManager()
    except Exception as e:
        print(f"❌ Error initializing camera: {e}")
        exit(1)

    tracker = FaceTracker(camera_manager=cam)
    tracker.start()

    try:
        while tracker.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        tracker.stop()
        cam.release()