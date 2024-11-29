import os
import shutil
import cv2
import threading

from plank_model import PlankModel

def create_and_clean_folder(folder_path):
    """Tạo thư mục nếu chưa tồn tại và xóa sạch nội dung nếu đã tồn tại."""
    if os.path.exists(folder_path):
        # Xóa toàn bộ nội dung thư mục
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

class Recording:
    def __init__(self):
        self.output_folder = "captured_images"
        create_and_clean_folder(self.output_folder)

        self.output_result_folder = "captured_results"
        create_and_clean_folder(self.output_result_folder)

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Không thể mở camera")
            exit()

        self.image_count = 0
        self.recording = False  # Biến điều khiển ghi hình
        self.thread = None

        self.model = PlankModel()

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

            # save result image
            result_image_path = os.path.join(self.output_result_folder, f"result_{self.image_count}.jpg")
            cv2.imwrite(result_image_path, result_image)

    def start_recording(self):
        if not self.recording:
            self.thread = threading.Thread(target=self.record)
            self.thread.start()
            print("Bắt đầu ghi hình...")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.thread.join()  # Chờ luồng kết thúc
            print("Dừng ghi hình.")

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recording = Recording()

    try:
        while True:
            req = input("Nhập lệnh (record/stop/exit): ").strip().lower()
            if req == "e":
                recording.stop_recording()
                break
            elif req == "r":
                recording.start_recording()
            elif req == "s":
                recording.stop_recording()
            else:
                print("Lệnh không hợp lệ. Hãy nhập 'record', 'stop', hoặc 'exit'.")
    finally:
        recording.release()
        print("Đã thoát chương trình.")
