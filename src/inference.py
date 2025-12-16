
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import time
import torch

WEIGHT_PATH = "runs/detect/train2/weights/best.pt"
YAML_PATH = "helmet-vest-and-boots-detection-8/data.yaml"


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
        ts = int(time.time())
        video_path = f"uploads/input_{ts}.mp4"
        out_path = f"uploads/output_{ts}.mp4"

        with open(video_path, "wb") as f:
            f.write(file.file.read())

        # ---------------- CUDA CHECK ----------------
        use_cuda = torch.cuda.is_available()
        device = 0 if use_cuda else "cpu"

        if use_cuda:
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA not available, using CPU")

        # ---------------- LOAD YOLO ----------------
        model = YOLO(model_path).to(0)

        # ---------------- VIDEO IO ----------------
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open uploaded video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None

        # ---------------- INFERENCE LOOP ----------------
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(
                frame,
                device=device,
                verbose=False
            )

            annotated = results[0].plot()

            if out is None:
                h, w, _ = annotated.shape
                out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            out.write(annotated)

        cap.release()
        if out:
            out.release()
        else:
            raise RuntimeError("No frames written to output video")

        return out_path

    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    

