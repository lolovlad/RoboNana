import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from math import ceil
from transformers import pipeline
from PIL import Image

DETECTION_COLOR = (0, 255, 0)
POSE_COLOR_1 = (245, 117, 66)
POSE_COLOR_2 = (245, 66, 230)
SAFE_ZONE_COLOR = (255, 0, 0)  # Red for the safe zone

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def draw_landmarks_on_image(rgb_image, pose_landmarks, crop_x, crop_y, crop_width, crop_height):
    global_landmarks = [
        landmark_pb2.NormalizedLandmark(
            x=(crop_x + landmark.x * crop_width) / rgb_image.shape[1],
            y=(crop_y + landmark.y * crop_height) / rgb_image.shape[0],
            z=landmark.z
        )
        for landmark in pose_landmarks.landmark
    ]

    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend(global_landmarks)

    mp_drawing.draw_landmarks(
            rgb_image,
            pose_landmarks_proto,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=POSE_COLOR_1, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=POSE_COLOR_2, thickness=2, circle_radius=2)
        )


def visualize_objects(rgb_image, detection_result):
    annotated_image = rgb_image

    for box in detection_result:
        start_point = int(box.xyxy[0][0]), int(box.xyxy[0][1])
        end_point = int(box.xyxy[0][2]), int(box.xyxy[0][3])
        cv2.rectangle(annotated_image, start_point, end_point, DETECTION_COLOR, 2)

        class_name = model.names[int(box.cls[0])]
        probability = ceil((box.conf[0] * 100)) / 100
        result_text = class_name + ' (' + str(probability) + ')'
        text_location = (start_point[0] + 5, start_point[1] + 15)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 1, DETECTION_COLOR, 2)

def draw_rectangle(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing, safe_zone_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
        x2, y2 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            x2, y2 = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y
        safe_zone_defined = True


def get_person(res):
    class_name = model.names[int(res.cls[0])]
    if class_name == "person":
        return True
    return False


def get_area_box(x1: int, y1: int, x2: int, y2: int) -> int:
    return (x2 - x1) * (y2 - y1)


def get_person_crop(frame, boxs, padding=200):
    new_box = []
    for box in boxs:
        start_point = int(box.xyxy[0][0]), int(box.xyxy[0][1])
        end_point = int(box.xyxy[0][2]), int(box.xyxy[0][3])
        area = get_area_box(*start_point, *end_point)
        new_box.append([start_point, end_point, area])
    max_area = max(new_box, key=lambda x: x[2])
    x1 = max_area[0][0] - padding if max_area[0][0] - padding > 0 else 1
    x2 = max_area[1][0] + padding if max_area[1][0] + padding < frame.shape[1] else frame.shape[1]
    y1 = max_area[0][1] - padding if max_area[0][1] - padding > 0 else 1
    y2 = max_area[1][1] + padding if max_area[1][1] + padding < frame.shape[0] else frame.shape[0]

    return frame[y1: y2,
                 x1: x2].copy(), [[x1, y1], [x2, y2], max_area[2]]


def get_depth_img(depth):
    depth_np = np.array(depth)

    depth_normalized = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    return depth_colormap


video_path = 'videokids.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Не удалось открыть видео.")
    exit()

# Initialize safe zone rectangle coordinates and drawing state
drawing = False
x1, y1, x2, y2 = 0, 0, 0, 0
safe_zone_defined = False

model = YOLO("yolo11n.pt")
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

cv2.namedWindow('main')
cv2.setMouseCallback('main', draw_rectangle)

new_width = 900
new_height = 500

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        depth = pipe(Image.fromarray(frame_rgb))["depth"]
        depth = get_depth_img(depth)

        result = model(frame, verbose=False)
        result = list(filter(get_person, result[0].boxes))
        if len(result) > 0:
            crop, local_box = get_person_crop(frame, result)

        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
        pose_detection_result = pose.process(mp_frame.numpy_view())
        pose_landmarks = pose_detection_result.pose_landmarks
        if pose_landmarks is not None:
            draw_landmarks_on_image(frame,
                                    pose_landmarks,
                                    local_box[0][0],
                                    local_box[0][1],
                                    local_box[1][0] - local_box[0][0],
                                    local_box[1][1] - local_box[0][1])
        visualize_objects(frame, result)

        if safe_zone_defined:
            cv2.rectangle(frame, (x1, y1), (x2, y2), SAFE_ZONE_COLOR, 2)

        cv2.imshow('main', frame)
        cv2.imshow('crop', crop)
        cv2.imshow('depth', depth)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

