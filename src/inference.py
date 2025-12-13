import yaml
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

WEIGHT_PATH = "runs/detect/train2/weights/best.pt"
YAML_PATH = "helmet-vest-and-boots-detection-8/data.yaml"

import os

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)





def run_inference_image(file,model_path):
    # Read image
    # Load model once
    try:
        model = YOLO(model_path)
        img_bytes = file.file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model(img)

        # Save output image
        out_path = f"uploads/result_{int(time.time())}.jpg"
        annotated = results[0].plot()
        cv2.imwrite(out_path, annotated)

        return out_path
    except Exception as e:
        return e


def run_inference_video(file, model_path):
    
    try:

        # Save uploaded video
        video_path = f"uploads/input_{int(time.time())}.mp4"
        out_path = f"uploads/output_{int(time.time())}.mp4"

        # Write file content to disk
        with open(video_path, "wb") as f:
            f.write(file.file.read())

        # Load YOLO model
        model = YOLO(model_path)

        # Open saved video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open uploaded video file")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO inference
            results = model(frame)
            annotated = results[0].plot()

            # Create output writer once dimensions known
            if out is None:
                h, w, _ = annotated.shape
                out = cv2.VideoWriter(out_path, fourcc, 20, (w, h))

            out.write(annotated)

        cap.release()

        # Release writer only if created
        if out is not None:
            out.release()
        else:
            raise RuntimeError("Video contains no frames or invalid file format")

        return out_path
    
    except Exception as e:
        return e
    

