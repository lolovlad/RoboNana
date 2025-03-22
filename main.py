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
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from mediapipe.framework.formats import landmark_pb2

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize Pose Landmarker (Do this *once* at the beginning)
model_path = 'pose_landmarker_full.task'  # Replace with the ACTUAL absolute path!
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in
            pose_landmarks
        ])
        mp_drawing.draw_landmarks(  # Use mp_drawing instead of solutions.drawing_utils
            annotated_image,
            pose_landmarks_proto,
            mp_pose.POSE_CONNECTIONS, # Use mp_pose.POSE_CONNECTIONS instead of solutions.pose.POSE_CONNECTIONS
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),  # Use mp_drawing here too
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
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

    # Convert the frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a MediaPipe image
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect pose landmarks
    detection_result = detector.detect(image)

    # Draw landmarks on the frame
    annotated_image = draw_landmarks_on_image(frame_rgb, detection_result)
    frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV


    cv2.imshow('Воспроизведение видео с ориентирами позы', frame)  # Изменен заголовок окна

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()