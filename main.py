import os
import shutil
import cv2
import threading
import time
import serial
from plank_model import PlankModel

def create_and_clean_folder(folder_path):
    """Tạo thư mục nếu chưa tồn tại và xóa sạch nội dung nếu đã tồn tại."""
    if os.path.exists(folder_path):
        # Xóa toàn bộ nội dung thư mục
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

class Recording:
    def __init__(self, serial_port):
        self.output_folder = "/home/nhutdeptrai/NCKH/gym-exercise-correction-tiny/captured_images"
        create_and_clean_folder(self.output_folder)

        self.output_result_folder = "/home/nhutdeptrai/NCKH/gym-exercise-correction-tiny/captured_results"
        create_and_clean_folder(self.output_result_folder)

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Không thể mở camera")
            exit()
        
        print("Đã mở camera")

        self.image_count = 0
        self.recording = False
        self.thread = None
        self.model = PlankModel()
        self.serial_port = serial_port  # Lưu đối tượng serial port để gửi phản hồi

    def record(self):
        self.recording = True
        while self.recording:
            ret, frame = self.cap.read()
            if not ret:
                print("Không thể nhận khung hình từ camera.")
                break

            image_path = os.path.join(self.output_folder, f"image_{self.image_count}.jpg")
            cv2.imwrite(image_path, frame)
            self.image_count += 1

            result_image = self.model.predict(frame=frame, prediction_probability_threshold=0.5)
            result_image_path = os.path.join(self.output_result_folder, f"result_{self.image_count}.jpg")
            cv2.imwrite(result_image_path, result_image)

            # Gửi tín hiệu "Oke" qua cổng serial sau khi dự đoán xong
            try:
                message = "Oke"
                self.serial_port.write(message.encode('utf-8'))
            except Exception as e:
                print(f"Lỗi khi gửi phản hồi qua UART: {e}")

    def start_recording(self):
        if not self.recording:
            self.thread = threading.Thread(target=self.record)
            self.thread.start()
            print("Bắt đầu ghi hình...")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.thread.join()
            print("Dừng ghi hình.")

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    serial_port = serial.Serial(
        port="/dev/ttyTHS1",
        baudrate=9600,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )
    time.sleep(1)

    recording = Recording(serial_port)
    
    try:
        while True:
            if serial_port.inWaiting() > 0:
                data = serial_port.read()
                decoded_data = data.decode('utf-8').strip()

                print(f"Received command: {decoded_data}")
                if decoded_data == "r":
                    recording.start_recording()
                elif decoded_data == "s":
                    recording.stop_recording()
                elif decoded_data == "e":
                    recording.stop_recording()
                    break
                else:
                    print("Lệnh không hợp lệ. Hãy nhập 'r', 's', hoặc 'e'.")
    except KeyboardInterrupt:
        print("Exiting Program")

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        recording.release()
        serial_port.close()
        print("Đã thoát chương trình.")
