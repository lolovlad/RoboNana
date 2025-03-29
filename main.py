'''
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image
import cv2

cap = None
current_position = 0
playing = False
fps = 30
video_path = None


def open_file():
    global cap, current_position, playing, fps, video_path
    file_path = filedialog.askopenfilename(filetypes=[('Video files', ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv'))])
    if file_path:
        print(f'Выбранный файл: {file_path}')
        video_path = file_path
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            cap = None
            return
        current_position = 0
        fps_value = int(cap.get(cv2.CAP_PROP_FPS))
        if fps_value > 0:
            fps = fps_value
        else:
            fps = 30

        playing = True
        update_frame()
        #play_button['state'] = 'normal'
        #rewind_button['state'] = 'normal'
        #forward_button['state'] = 'normal'


def update_frame():
    global cap, current_position, playing, video_path
    if cap is not None and playing:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_position)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (720, 480))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            label.config(image=image)
            label.image = image

            current_position += 1
            label.after(int(1000 / fps), update_frame)
        else:
            stop_video()


def stop_video():
    global cap, playing
    playing = False
    if cap is not None:
        cap.release()
        cap = None
    #play_button['state'] = 'normal'
    #rewind_button['state'] = 'normal'
    #forward_button['state'] = 'normal'


def start_video():
    global cap, playing, video_path, current_position

    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            cap = None
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_position)
        playing = True
        update_frame()
        play_button['state'] = 'normal'
        rewind_button['state'] = 'normal'
        forward_button['state'] = 'normal'


def rewind_video():
    global current_position, cap
    if cap is not None:
        current_position = max(0, current_position - fps * 10)
        update_frame()
def forward_video():
    global current_position, cap
    if cap is not None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_position = min(total_frames - 1, current_position + fps * 10)
        update_frame()
        
def toggle_playback():
    global cap, playing, video_path, current_position
    if playing:
        stop_video()
        play_button.config(text='Воспроизвести')
    else:
        if video_path is not None:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Could not open video file.")
                cap = None
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_position)
            playing = True
            update_frame()
            play_button.config(text='Остановить')

root = tk.Tk()
root.title('Камера')

label = tk.Label(root)
label.pack()

open_button = ttk.Button(root, text='Открыть файл', command=open_file)
open_button.pack()

frame_buttons = ttk.Frame(root)
frame_buttons.pack()


play_button = ttk.Button(frame_buttons, text='Остановить', command=toggle_playback)
play_button.pack(side='left', padx=5)

rewind_button = ttk.Button(frame_buttons, text='Перемотать << 10 сек', command=rewind_video, state='disabled')
rewind_button.pack(side='left', padx=5)

forward_button = ttk.Button(frame_buttons, text='Перемотать >> 10 сек', command=forward_video, state='disabled')
forward_button.pack(side='left', padx=5)


root.mainloop()
'''

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DETECTION_COLOR = (0, 255, 0)
POSE_COLOR_1 = (245, 117, 66)
POSE_COLOR_2 = (245, 66, 230)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose_model_path = 'pose_landmarker_full.task'
pose_base_options = python.BaseOptions(model_asset_path=pose_model_path)
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    output_segmentation_masks=True)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

object_model_path = 'efficientdet_lite0.tflite' # Make sure this is in the correct path
object_base_options = python.BaseOptions(model_asset_path=object_model_path)
object_options = vision.ObjectDetectorOptions(base_options=object_base_options,
                                       score_threshold=0.5)
object_detector = vision.ObjectDetector.create_from_options(object_options)


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in
            pose_landmarks
        ])
        mp_drawing.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=POSE_COLOR_1, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=POSE_COLOR_2, thickness=2, circle_radius=2)
        )
    return annotated_image

def visualize_objects(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)

    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, DETECTION_COLOR, 2)

        category = detection.categories[0]
        class_name = category.category_name
        probability = round(category.score, 2)
        result_text = class_name + ' (' + str(probability) + ')'
        text_location = (bbox.origin_x + 5, bbox.origin_y + 15)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 1, DETECTION_COLOR, 2)

    return annotated_image

video_path = 'videokids2.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    pose_detection_result = pose_detector.detect(mp_image)
    object_detection_result = object_detector.detect(mp_image)

    annotated_image = draw_landmarks_on_image(frame_rgb, pose_detection_result) # Draw pose landmarks
    annotated_image = visualize_objects(annotated_image, object_detection_result) # Draw object detections
    frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR) # Back to BGR for OpenCV


    cv2.imshow('Воспроизведение видео с ориентирами позы и обнаружением объектов', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
