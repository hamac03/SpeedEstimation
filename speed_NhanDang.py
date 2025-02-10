from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
import time
from paddleocr import PaddleOCR
import torch
from realesrgan import RealESRGAN
from PIL import Image
from format import format_1_line
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def clean_result(result):
    # Tìm tất cả các ký tự in hoa và số từ chuỗi kết quả
    cleaned_result = ''.join(re.findall(r'[A-Z0-9]', result))
    return cleaned_result

def perform_ocr(ocr, image):
    try:
        res = ocr.ocr(image)
        if not res:
            return "No text detected"
        values = [item[1][0] for sublist in res for item in sublist]
        if len(values) == 2:
            result = ''
            for line in values:
                for char in line:
                    if char.isalnum():
                        result += char
                result += '-'
        elif len(values) == 1:
            result = ''
            for line in values:
                for char in line:
                    if char.isalnum():
                        result += char
        else:
            result = '-'.join(values)
        result = result.strip('-')
        print(values)
        print(result)
        # return result
        # return format_1_line(result)
        return clean_result(format_1_line(result))
    except Exception as e:
        return "No text detected"

model = RealESRGAN(device, scale=2)
model.load_weights('Real-ESRGAN-colab/weights/RealESRGAN_x2.pth')

ocr = PaddleOCR(lang='en')
vehicle_detection_model = YOLO('model/vehicle_detection.pt')
vehicle_detection_model.to('cuda')
np_model = YOLO('model/best_np.pt')
np_model.to('cuda')
# video_path = "data/IMG_1389.MOV"
video_path = "data/4k_format_20.mp4"
cap = cv2.VideoCapture(video_path)

colors = [(0, 128, 255),  # cam
            (0, 255, 0),  # Xanh lá cây
            (127, 127, 127),  # Vàng
            (255, 165, 0)]  # Xanh nhạt
speeds_track = {}

offset = 10
speed_thres = 80
up = {}
speed_dict = {}
counter_up = []

dict_track_lp = {}
dict_plate = {}

distance_2_lines = 28
# red_line_x_min = 200
# red_line_x_max = 2400
# red_line_y = 900
# blue_line_y = 1300

red_line_x_min = 300
red_line_x_max = 3600
red_line_y = 1350
blue_line_y = 1950
line_tracking_thresh = 700

map_class = {0: 'bus', 1: 'car', 2: 'motorbike', 3: 'truck'}
speed_thresh = 80
index = 0
text_color = (0, 0, 0)  # Black color for text
yellow_color = (0, 255, 255)  # Yellow color for background
red_color = (0, 0, 255)  # Red color for lines
blue_color = (255, 0, 0)  # Blue color for lines
fps_start_time = 0
fps = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print('The video capture end.')
        break
    results = vehicle_detection_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.4)
    frame_cop = frame.copy()
    # filtered_boxes = []
    # filtered_track_ids = []

    for detection in results[0].boxes.data.tolist():
        x1, y1, x2, y2, track_id, score, class_id = detection
        x3 = int(x1)
        y3 = int(y1)
        x4 = int(x2)
        y4 = int(y2)
        if line_tracking_thresh > y4:
            continue
        track_id = int(track_id)
        class_id = int(class_id)
        #Tính toán toạ độ trung điểm dưới của bounding box
        cx = int(x3 + x4) // 2
        cy = y4
        cv2.circle(frame_cop, (cx, cy), 4, (0, 0, 255), -1)
        # Nếu xe đi qua khu vực tính toán tốc độ và tốc độ xe vượt qua giới hạn cho phép thì hiển bị bounding box màu đỏ cùng với tốc độ
        if y4 < red_line_y and track_id in speed_dict and speed_dict[track_id] > speed_thres:
            cv2.rectangle(frame_cop, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cv2.putText(frame_cop, f"#{track_id} - {map_class[class_id]}", (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)


        else:
            cv2.rectangle(frame_cop, (x3, y3), (x4, y4), colors[class_id], 2)  # Draw bounding box
            cv2.putText(frame_cop, f"#{track_id} - {map_class[class_id]}", (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 1.5, colors[class_id], 2)

        if track_id in dict_plate:
            cv2.putText(frame_cop, dict_plate[track_id], (x3, y4), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                        colors[class_id], 2)

        # Kiểm tra xem lân toạ độ của trung điểm dưới của y có nằm trong vùng bắt đầu tính thời gian (start line) hay không
        if (cy + offset) > blue_line_y > (cy - offset):
            # Nếu xe đó chưa nằm trong danh sách thì khởi tạo xe đó với thời gian vào vùng đếm là 0s
            if track_id not in up:
                up[track_id] = 0  # Khởi tạo biến đếm cho đối tượng
                continue

        # Nếu xe đã vào vùng tính thời gian
        if track_id in up:
            # Nếu xe đến vùng ngưng tính số frame
            if (cy + offset) > red_line_y > (cy - offset):
                # Tính thời gian bằng cách lấy số frame / fps của video
                elapsed1_time = up[track_id] / 30
                # Thêm xe vào dánh sách đếm xe khi xe đã đi qua vùng kết thúc.
                distance1 = distance_2_lines
                if elapsed1_time != 0:
                    a_speed_ms1 = distance1 / elapsed1_time
                else:
                    a_speed_ms1 = 0
                # Đổi từ m/s -> km/h
                a_speed_kh1 = a_speed_ms1 * 3.6
                a_speed_kh1 = round(a_speed_kh1, 2)
                # Lưu lại tốc độ của xe đó và vẽ lên màn hình.
                speed_dict[track_id] = a_speed_kh1
                cv2.putText(frame_cop, f"{up[track_id] / 30:.2f} s - {(a_speed_kh1)} km/h", (x4, y4 - 20), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                            (0, 255, 255), 2)
                continue

        # Nếu xe đã qua vùng màu đỏ và tốc độ xe có tồn tại trong speed_dict và giá tri đó không phaỉ None thì vẽ lên màn hình
        if red_line_y > (cy - offset + 10) and track_id in speed_dict and speed_dict[track_id] is not None:
            cv2.putText(frame_cop, str(speed_dict[track_id]) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                        (0, 255, 255), 2)
            continue
        # Nếu xe ở giữa 2 đường thì hiển thị
        if track_id in up:
            # cộng số frame lên.
            up[track_id] += 1
            # Phát hiện và nhận diện biển số xe
            if track_id in dict_plate:
                cv2.putText(frame, dict_plate[track_id], (x3, y4),  cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
                continue
            vehicle_bounding_boxes = []
            vehicle_bounding_boxes.append([x3, y3, x4, y4, track_id, score])

            for bbox in vehicle_bounding_boxes:
                # if track_id in dict_plate:
                #     continue
                # license plate detector for region of interest
                roi = frame[y3:y4, x3:x4]
                license_plates = np_model(roi)[0]


                # check every bounding box for a license plate
                for license_plate in license_plates.boxes.data.tolist():
                    plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate
                    # verify detections
                    # print(license_plate, 'track_id: ' + str(bbox[4]))
                    plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                    print(plate.shape)
                    try:
                        # Chuyển đổi ảnh từ BGR sang RGB
                        img = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)

                        # Chuyển đổi ảnh từ định dạng numpy array của OpenCV sang đối tượng PIL.Image
                        img = Image.fromarray(img)
                        sr_image = model.predict(img)

                        sr_image = np.array(sr_image)
                        print("shape org", sr_image.shape)
                        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
                        # cv2.imshow('sr', sr_image)
                        result_ocr = perform_ocr(ocr, sr_image)
                    except Exception as e:
                        print("Error occurred while processing image:", e)
                        result_ocr = perform_ocr(ocr, plate)
                    print(result_ocr)
                    dict_plate[track_id] = result_ocr
                    # cv2.imshow(f'plate{track_id}_{plate_y2}', plate)
                    # cv2.imwrite(f"output/plate{video_choose}/{track_id}.jpg", plate)
                    plate_x1 = int(plate_x1)
                    plate_y1 = int(plate_y1)
                    plate_x2 = int(plate_x2)
                    plate_y2 = int(plate_y2)
                    # Adjust the coordinates to match the original frame
                    original_plate_x1 = x3 + plate_x1
                    original_plate_y1 = y3 + plate_y1
                    original_plate_x2 = x3 + plate_x2
                    original_plate_y2 = y3 + plate_y2
                    # Draw the license plate rectangle on the original frame
                    cv2.rectangle(frame, (original_plate_x1, original_plate_y1), (original_plate_x2, original_plate_y2),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, result_ocr, (original_plate_x1, original_plate_y1),  cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

            # Hiển thị số thời gian đi bằng cách chia số frame / fps
            cv2.putText(frame_cop, f"{up[track_id]/30:.2f} s", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0, 255, 255), 2)

    # Kẻ 2 đường màu đó lên màn hình
    cv2.line(frame_cop, (red_line_x_min, red_line_y), (red_line_x_max, red_line_y), red_color, 2)
    cv2.putText(frame_cop, 'End Line', (red_line_x_min, red_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 5, cv2.LINE_AA)
    cv2.line(frame_cop, (0, blue_line_y), (3839, blue_line_y), blue_color, 2)
    cv2.putText(frame_cop, 'Start Line', (0, blue_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 5, cv2.LINE_AA)
    cv2.line(frame_cop, (0, line_tracking_thresh), (3839, line_tracking_thresh), red_color, 2)


    #calculate fps
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1 / time_diff
    fps_start_time = fps_end_time

    cv2.putText(frame_cop, f"FPS:"
                         f" {fps:.2f}", (5, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 2)
    frame_cop = cv2.resize(frame_cop, (1366, 768))
    cv2.imshow('tracking', frame_cop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()