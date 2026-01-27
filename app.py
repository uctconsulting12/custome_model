from src.model import train_yolo_model
from src.file import unzip_and_locate,parse_classes,create_yaml
TRAIN_RATIO = 0.8
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
import time


from fastapi import FastAPI,UploadFile, File,Form,Request,HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import subprocess
import threading
from fastapi.staticfiles import StaticFiles

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

HLS_ROOT = "hls_out"
os.makedirs(HLS_ROOT, exist_ok=True)

# Serve HLS output
app.mount("/hls", StaticFiles(directory=HLS_ROOT), name="hls")

# Keep track of streams
streams = {} 


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
async def start_training(request: TrainRequest):
    try:
        results = await train_yolo_model(request)
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






HLS_ROOT = "hls_out"
os.makedirs(HLS_ROOT, exist_ok=True)

# Serve HLS folders
app.mount("/hls", StaticFiles(directory=HLS_ROOT), name="hls")

# Load YOLO once (GPU will be used if available)
model = YOLO("yolov8n.pt")

# Keep track of running streams
streams = {}  # stream_id -&gt; {"thread":..., "stop": threading.Event(), "proc":..., "out_dir":...}

class StartRequest(BaseModel):
    source: str              # RTSP URL or file path
    use_nvenc: bool = True


class StartRequest(BaseModel):
    source: str          # RTSP / file / HTTP stream
    weights_path: str
    camera_id: str
    user_id: str
    org_id: str
    use_nvenc: bool = False


def stream_worker(
    stream_id: str,
    req: StartRequest,
    stop_event: threading.Event
):
    out_dir = os.path.join(HLS_ROOT, stream_id)
    os.makedirs(out_dir, exist_ok=True)
    out_m3u8 = os.path.join(out_dir, "stream.m3u8")

    model = YOLO(req.weights_path)

    cap = cv2.VideoCapture(req.source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    gop = int(fps * 2)

    vcodec = "h264_nvenc" if req.use_nvenc else "libx264"

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",

        "-c:v", vcodec,
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-g", str(gop),
        "-keyint_min", str(gop),
        "-sc_threshold", "0",

        "-f", "hls",
        "-hls_time", "4",
        "-hls_list_size", "10",
        "-hls_flags", "delete_segments+append_list",
        out_m3u8
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    streams[stream_id]["proc"] = proc

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            annotated = results[0].plot()

            proc.stdin.write(annotated.tobytes())

    except BrokenPipeError:
        pass
    finally:
        cap.release()
        try:
            proc.stdin.close()
            proc.terminate()
        except Exception:
            pass



@app.post("/streams/start")
def start_stream(req: StartRequest):
    stream_id = str(uuid.uuid4())
    stop_event = threading.Event()

    streams[stream_id] = {
        "stop": stop_event,
        "thread": None,
        "proc": None
    }

    t = threading.Thread(
        target=stream_worker,
        args=(stream_id, req, stop_event),
        daemon=True
    )
    streams[stream_id]["thread"] = t
    t.start()

    time.sleep(15)

    return {
        "stream_id": stream_id,
        "camera_id": req.camera_id,
        "user_id": req.user_id,
        "org_id": req.org_id,
        "hls_url": f"/hls/{stream_id}/stream.m3u8"
    }



@app.post("/streams/stop/{stream_id}")
def stop_stream(stream_id: str):
    if stream_id not in streams:
        raise HTTPException(status_code=404, detail="stream not found")

    streams[stream_id]["stop"].set()

    proc = streams[stream_id].get("proc")
    if proc:
        try:
            proc.terminate()
        except Exception:
            pass

    dir_path = os.path.join(HLS_ROOT, stream_id)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)

    streams.pop(stream_id, None)

    return {"status": "stopped", "stream_id": stream_id}
