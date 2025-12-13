import threading
import uuid
import time
from ultralytics import YOLO
from pathlib import Path

# Store all jobs and logs
JOB_STATUS = {}
JOB_LOGS = {}

def append_log(job_id, msg):
    JOB_LOGS[job_id].append(msg)
    print(msg)   # Also show in console



def run_training(job_id, params):
    try:
        append_log(job_id, f"[{job_id}] Training started…")

        model = YOLO(params["model_name"])

        # Training loop
        results = model.train(
            data=params["data_yaml"],
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            device=params["device"],
            batch=params.get("batch", 16),
            lr0=params.get("lr", 0.001)
        )

        save_dir = str(results.save_dir)

        JOB_STATUS[job_id] = {
            "status": "completed",
            "save_dir": save_dir
        }

        append_log(job_id, f"[{job_id}] Training Completed! Output: {save_dir}")

    except Exception as e:
        JOB_STATUS[job_id] = {"status": "error", "error": str(e)}
        append_log(job_id, f"[{job_id}] ERROR: {str(e)}")


def start_training_job(params):
    job_id = str(uuid.uuid4())

    JOB_STATUS[job_id] = {"status": "running"}
    JOB_LOGS[job_id] = []

    thread = threading.Thread(target=run_training, args=(job_id, params))
    thread.start()

    return job_id
