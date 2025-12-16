from src.model import train_yolo_model
from src.file import unzip_and_locate,parse_classes,create_yaml
TRAIN_RATIO = 0.8
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


from fastapi import FastAPI,UploadFile, File,Form,Request
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

from typing import Optional, Dict, Any, List, Union

from fastapi.responses import JSONResponse,StreamingResponse
import zipfile
import os
import shutil
import random
import yaml
import uuid
import json
import base64
from ultralytics import YOLO
import requests
import cv2
import asyncio


from src.inference import run_inference_image, run_inference_video
from src.database import insert_model_details,get_models_by_org_user


app = FastAPI(title="YOLO Training API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.post("/prepare-dataset")
async def prepare_dataset(
    images_zip: UploadFile,
    labels_zip: UploadFile,
    classes: str = Form(...)
):
    try:
        class_list = parse_classes(classes)

        job_id = str(uuid.uuid4())
        base_dir = f"dataset/dataset_{job_id}"
        unzip_dir = os.path.join(base_dir, "unzipped")
        os.makedirs(unzip_dir, exist_ok=True)

        images_path = os.path.join(base_dir, "images.zip")
        labels_path = os.path.join(base_dir, "labels.zip")

        with open(images_path, "wb") as f:
            f.write(await images_zip.read())
        with open(labels_path, "wb") as f:
            f.write(await labels_zip.read())

        img_root = unzip_and_locate(images_path, os.path.join(unzip_dir, "images"))
        lbl_root = unzip_and_locate(labels_path, os.path.join(unzip_dir, "labels"))

        # Output folders
        train_img = os.path.join(base_dir, "train/images")
        train_lbl = os.path.join(base_dir, "train/labels")
        val_img = os.path.join(base_dir, "valid/images")
        val_lbl = os.path.join(base_dir, "valid/labels")
        os.makedirs(train_img, exist_ok=True)
        os.makedirs(train_lbl, exist_ok=True)
        os.makedirs(val_img, exist_ok=True)
        os.makedirs(val_lbl, exist_ok=True)

        image_files = [f for f in os.listdir(img_root) if f.lower().endswith(IMAGE_EXT)]
        label_files = [f for f in os.listdir(lbl_root) if f.lower().endswith(".txt")]

        paired = [(img, os.path.splitext(img)[0] + ".txt") for img in image_files if os.path.splitext(img)[0] + ".txt" in label_files]

        random.shuffle(paired)
        train_count = int(len(paired) * TRAIN_RATIO)

        for img, lbl in paired[:train_count]:
            shutil.move(os.path.join(img_root, img), os.path.join(train_img, img))
            shutil.move(os.path.join(lbl_root, lbl), os.path.join(train_lbl, lbl))

        for img, lbl in paired[train_count:]:
            shutil.move(os.path.join(img_root, img), os.path.join(val_img, img))
            shutil.move(os.path.join(lbl_root, lbl), os.path.join(val_lbl, lbl))

        create_yaml(base_dir, class_list)

        return JSONResponse({
            "status": "success",
            "output_dir": base_dir,
            "train_samples": train_count,
            "valid_samples": len(paired) - train_count,
            "classes": class_list,
            "yml_path":f"{base_dir}/data.yaml"
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)




# -------- Request Body Schema ---------
class TrainRequest(BaseModel):
    org_id:int
    user_id:int
    name:str
    model_name: str = Field(default="yolo11n.pt")
    data_yaml: str 
    epochs: int = Field(default=50)
    imgsz: int = Field(default=640)

    # device can be: 0, "0", "0,1", [0,1]
    device: Union[int, str, List[int]] = Field(default="0")

    batch: int = Field(default=8)
    lr: float = Field(default=0.001)
    

def format_results(results):
    metrics_dict = results.results_dict
    precision, recall, map50, map5095 = results.mean_results()

    save_dir = results.save_dir
    
    best_weight = os.path.join(save_dir, "weights", "best.pt")
    last_weight = os.path.join(save_dir, "weights", "last.pt")

    return {
        "status": "success",
        "message": "Training completed",
        "weights_path": {
            "best": best_weight,
            "last": last_weight
        },
        "summary": {
            "metrics": {
                "precision": precision,
                "recall": recall,
                "mAP50": map50,
                "mAP50-95": map5095,
                "raw": metrics_dict
            },
            "dataset": {
                "num_images": int(results.nt_per_class.sum()),
                "num_classes": len(results.names),
                "classes": results.names
            },
            "speed_ms": results.speed,
        },
    }






# ------------- API Endpoint ----------------
@app.post("/train")
def start_training(request: TrainRequest):
    try:
        results = train_yolo_model(request)
        format_result=format_results(results)
        model_weight=format_result["weights_path"]["best"]
        data={
            "org_id":request.org_id,
            "user_id":request.user_id,
            "name":request.name,
            "model_weight_path":model_weight
        }
        insert_model_details(data)
        return {
            "status": "success",
            "message": "Training completed",
            "training": "training process completed successfully..",
            "matrics": format_result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    


    
class input_parameter(BaseModel):
    org_id:int
    user_id:int

@app.post("/custom-model")
def custom_model_details(req:input_parameter):
    model_details=get_models_by_org_user(req.org_id,req.user_id)
    return {"model_details":model_details}
       

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...),model_path: str = Form(...),
    ):
    result_path = run_inference_image(file,model_path)
    return FileResponse(result_path, media_type="image/jpeg")



@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...),model_path: str = Form(...),
   ):
    result_path = run_inference_video(file,model_path)
    return FileResponse(result_path, media_type="video/mp4")




# ---------------------------------------------------
# Pydantic model (input fields)
# ---------------------------------------------------
class StreamRequest(BaseModel):
    weights_path: str
    video_url: str
    camera_id: int
    user_id: int
    org_id: int


# Cache models
model_cache = {}

def get_model(weights_path: str):
    if weights_path not in model_cache:
        model_cache[weights_path] = YOLO(weights_path).to("cuda")
    return model_cache[weights_path]


# -------------------------
# Request Model
# -------------------------
class StreamRequest(BaseModel):
    weights_path: str
    video_url: str
    camera_id: int
    user_id: int
    org_id: int


# -------------------------
# GPU Model Cache
# -------------------------
model_cache = {}

def get_model(weights_path):
    if weights_path not in model_cache:
        model_cache[weights_path] = YOLO(weights_path)
        print("cuda" if torch.cuda.is_available() else "cpu")
        model_cache[weights_path].to("cuda" if torch.cuda.is_available() else "cpu")
    return model_cache[weights_path]


# ----------------------------------------------------
# YOLO Streaming Endpoint (NDJSON)
# ----------------------------------------------------
@app.post("/stream_yolo")
async def stream_yolo(req: StreamRequest):

    weights_path = req.weights_path
    video_url = req.video_url

    camera_id = req.camera_id
    user_id = req.user_id
    org_id = req.org_id

    model = get_model(weights_path)

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        return {"error": "Cannot open video URL"}

    async def generate():
        buffer = ""

        while True:

            ret, frame = cap.read()
            if not ret:
                print("Stream ended")
                break

            # Run YOLO inference
            results = model.predict(frame, imgsz=640, conf=0.5, device=0, verbose=False)

            detections = []
            for r in results:
                for box in r.boxes:
                    detections.append({
                        "class": model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy.tolist()[0]
                    })

            # Annotate frame
            annotated = results[0].plot()
            ret2, buffer_image = cv2.imencode(".jpg", annotated)
            if not ret2:
                continue

            frame_b64 = base64.b64encode(buffer_image).decode()

            # NDJSON FRAME
            payload = {
                "camera_id": camera_id,
                "user_id": user_id,
                "org_id": org_id,
                "detections": detections,
                "frame": frame_b64
            }

            json_line = json.dumps(payload) + "\n"

            # Send full JSON message
            yield json_line.encode()

            # Force flush
            await asyncio.sleep(0)

    return StreamingResponse(generate(), media_type="application/x-ndjson")