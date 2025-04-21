import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, DPTForDepthEstimation

MODEL_NAME = "Intel/dpt-hybrid-midas"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = DPTForDepthEstimation.from_pretrained(MODEL_NAME).to(DEVICE)


def estimate_depth(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth_cv = cv2.cvtColor(np.array(depth), cv2.COLOR_RGB2BGR)

    return depth_cv


video_path = 'videokids.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    depth_map = estimate_depth(frame)
    combined_image = np.concatenate((frame, depth_map), axis=1)
    cv2.imshow('Видео с глубиной', combined_image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()