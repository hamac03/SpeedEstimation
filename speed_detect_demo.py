from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
# from streamlit_file_browser import st_file_browser
import time
from paddleocr import PaddleOCR
import torch
from realesrgan import RealESRGAN
from PIL import Image
from ocr_utils import perform_ocr
from utils import clean_result
import os
from io import BytesIO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(DEVICE, scale=2)
model.load_weights('Real-ESRGAN-colab/weights/RealESRGAN_x2.pth')

ocr = PaddleOCR(lang='en')
vehicle_detection_model = YOLO('model/vehicle_detection.pt')
vehicle_detection_model.to('cuda')
np_model = YOLO('model/best_np.pt')
np_model.to('cuda')
# VIDEO_PATH = "data/IMG_1389.MOV"
# video_path = "data/4k_format_20.mp4"
# cap = cv2.VideoCapture(VIDEO_PATH)

# Constants
COLORS = [(0, 128, 255), (0, 255, 0), (127, 127, 127), (255, 165, 0)]
OFFSET = 12
DISTANCE_2_LINES = 28
RED_LINE_X_MIN = 300
RED_LINE_X_MAX = 3600
RED_LINE_Y = 1350
BLUE_LINE_Y = 1950
LINE_TRACKING_THRESHOLD = 700
MAP_CLASS = {0: 'bus', 1: 'car', 2: 'motorbike', 3: 'truck'}
SPEED_THRESHOLD = 80
TEXT_COLOR = (0, 0, 0)  # Black color for text
YELLOW_COLOR = (0, 255, 255)  # Yellow color for background
RED_COLOR = (0, 0, 255)  # Red color for lines
BLUE_COLOR = (255, 0, 0)  # Blue color for lines
GREEN_COLOR = (0, 255, 0)
VEHICLE_CONFIDENCE_THRESHOLD = 0.4
VEHICLE_TRACKER_CONFIG = "bytetrack.yaml"
TEXT_FONT = cv2.FONT_HERSHEY_COMPLEX
temp_video_path = ''

# Variables
speeds_track = {}
up = {}
speed_dict = {}
counter_up = []
# dict_track_lp = {}
dict_plate = {}
index = 0
fps_start_time = 0
fps = 0

video_types = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm', 'mpeg']
st.header('Hệ thống giám sát tốc độ')
st.sidebar.header('Chọn video để xử lý')
VIDEO = st.sidebar.file_uploader('Upload Video', type=video_types)
start_button = st.sidebar.button("Start")
# event = st_file_browser(label='Choose a video', start_path='.', key='video')
# st.write(event)
if start_button:
    if VIDEO is None:
        st.write('Please choose a video file')
    else:
        video_bytes = VIDEO.read()
        # Use OpenCV to capture video from byte stream
        temp_video_path = f"temp_{VIDEO.name}"

        # Save bytes to a temporary file
        with open(temp_video_path, 'wb') as temp_file:
            temp_file.write(video_bytes)

        # Use OpenCV to capture video from the temporary file
        cap = cv2.VideoCapture(temp_video_path)

        # VIDEO_PATH = 'data/' + VIDEO.name
        # cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            st.error("Error opening video file.")
        stframe = st.empty()
        stop_button = st.button("Stop")
        st.markdown(
            """
            <style>
            [data-testid="stImage"] {
                width: 100%;
                height: 100%;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()

            if not ret:
                print('The video capture end.')
                break
            results = vehicle_detection_model.track(frame, persist=True, tracker=VEHICLE_TRACKER_CONFIG, conf=VEHICLE_CONFIDENCE_THRESHOLD)
            frame_cop = frame.copy()
            # filtered_boxes = []
            # filtered_track_ids = []

            for detection in results[0].boxes.data.tolist():
                x1, y1, x2, y2, track_id, score, class_id = detection
                x3 = int(x1)
                y3 = int(y1)
                x4 = int(x2)
                y4 = int(y2)
                if LINE_TRACKING_THRESHOLD > y4:
                    continue
                track_id = int(track_id)
                class_id = int(class_id)
                #Tính toán toạ độ trung điểm dưới của bounding box
                cx = int(x3 + x4) // 2
                cy = y4
                cv2.circle(frame_cop, (cx, cy), 4, RED_COLOR, -1)
                # Nếu xe đi qua khu vực tính toán tốc độ và tốc độ xe vượt qua giới hạn cho phép thì hiển bị bounding box màu đỏ cùng với tốc độ
                if y4 < RED_LINE_Y and track_id in speed_dict and speed_dict[track_id] > SPEED_THRESHOLD:
                    cv2.rectangle(frame_cop, (x3, y3), (x4, y4), RED_COLOR, 2)
                    cv2.putText(frame_cop, f"#{track_id} - {MAP_CLASS[class_id]}", (x3, y3), TEXT_FONT, 1.5, RED_COLOR, 2)


                else:
                    cv2.rectangle(frame_cop, (x3, y3), (x4, y4), COLORS[class_id], 2)  # Draw bounding box
                    cv2.putText(frame_cop, f"#{track_id} - {MAP_CLASS[class_id]}", (x3, y3), TEXT_FONT, 1.5, COLORS[class_id], 2)

                if track_id in dict_plate:
                    cv2.putText(frame_cop, dict_plate[track_id], (x3, y4), TEXT_FONT, 1.5,
                                COLORS[class_id], 2)

                # Kiểm tra xem lân toạ độ của trung điểm dưới của y có nằm trong vùng bắt đầu tính thời gian (start line) hay không
                if (cy + OFFSET) > BLUE_LINE_Y > (cy - OFFSET):
                    # Nếu xe đó chưa nằm trong danh sách thì khởi tạo xe đó với thời gian vào vùng đếm là 0s
                    if track_id not in up:
                        up[track_id] = 0  # Khởi tạo biến đếm cho đối tượng
                        continue

                # Nếu xe đã vào vùng tính thời gian
                if track_id in up:
                    # Nếu xe đến vùng ngưng tính số frame
                    if (cy + OFFSET) > RED_LINE_Y > (cy - OFFSET):
                        # Tính thời gian bằng cách lấy số frame / fps của video
                        elapsed1_time = up[track_id] / 30
                        # Thêm xe vào dánh sách đếm xe khi xe đã đi qua vùng kết thúc.
                        if counter_up.count(track_id) == 0:
                            counter_up.append(track_id)
                            distance1 = DISTANCE_2_LINES
                            if elapsed1_time != 0:
                                a_speed_ms1 = distance1 / elapsed1_time
                            else:
                                a_speed_ms1 = 0
                            # Đổi từ m/s -> km/h
                            a_speed_kh1 = a_speed_ms1 * 3.6
                            a_speed_kh1 = round(a_speed_kh1, 2)
                            # Lưu lại tốc độ của xe đó và vẽ lên màn hình.
                            speed_dict[track_id] = a_speed_kh1
                            cv2.putText(frame_cop, f"{up[track_id] / 30:.2f} s - {(a_speed_kh1)} km/h", (x4, y4 - 20), TEXT_FONT, 1.5,
                                        YELLOW_COLOR, 2)
                        continue

                # Nếu xe đã qua vùng màu đỏ và tốc độ xe có tồn tại trong speed_dict và giá tri đó không phải None thì vẽ lên màn hình
                if RED_LINE_Y > (cy - OFFSET + 10) and track_id in speed_dict and speed_dict[track_id] is not None:
                    cv2.putText(frame_cop, str(speed_dict[track_id]) + 'Km/h', (x4, y4), TEXT_FONT, 1.5,
                                YELLOW_COLOR, 2)
                    continue
                # Nếu xe ở giữa 2 đường thì hiển thị
                if track_id in up:
                    # cộng số frame lên.
                    up[track_id] += 1
                    # Phát hiện và nhận diện biển số xe
                    if track_id in dict_plate:
                        cv2.putText(frame, dict_plate[track_id], (x3, y4),  TEXT_FONT, 1.5, GREEN_COLOR, 2)
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
                                        RED_COLOR, 2)
                            cv2.putText(frame, result_ocr, (original_plate_x1, original_plate_y1),  TEXT_FONT, 2, GREEN_COLOR, 2)

                    # Hiển thị số thời gian đi bằng cách chia số frame / fps
                    cv2.putText(frame_cop, f"{up[track_id]/30:.2f} s", (x4, y4), TEXT_FONT, 2.5, YELLOW_COLOR, 2)

            # Kẻ 2 đường màu đó lên màn hình
            cv2.line(frame_cop, (RED_LINE_X_MIN, RED_LINE_Y), (RED_LINE_X_MAX, RED_LINE_Y), RED_COLOR, 2)
            cv2.putText(frame_cop, 'End Line', (RED_LINE_X_MIN, RED_LINE_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 5, cv2.LINE_AA)
            cv2.line(frame_cop, (0, BLUE_LINE_Y), (3839, BLUE_LINE_Y), BLUE_COLOR, 2)
            cv2.putText(frame_cop, 'Start Line', (0, BLUE_LINE_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, TEXT_COLOR, 5, cv2.LINE_AA)
            cv2.line(frame_cop, (0, LINE_TRACKING_THRESHOLD), (3839, LINE_TRACKING_THRESHOLD), RED_COLOR, 2)


            #calculate fps
            fps_end_time = time.time()
            time_diff = fps_end_time - fps_start_time
            fps = 1 / time_diff
            fps_start_time = fps_end_time

            cv2.putText(frame_cop, f"FPS:"
                                f" {fps:.2f}", (5, 70), TEXT_FONT, 2, YELLOW_COLOR, 2)
            # frame_cop = cv2.resize(frame_cop, (1920, 1080))
            frame_cop = cv2.resize(frame_cop, (16*90, 9*90))
            # cv2.imshow('tracking', frame_cop)
            stframe.image(frame_cop, channels="BGR")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"Deleted temporary video file: {temp_video_path}")
        else:
            print(f"Temporary video file not found at: {temp_video_path}")