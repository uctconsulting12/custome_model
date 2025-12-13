# main.py
import threading
import uuid
import asyncio
import logging
import sys
import io
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from ultralytics import YOLO

# -----------------------------
# GLOBAL STORAGE & LOCKS
# -----------------------------
JOB_STATUS = {}   # job_id -> status dict
JOB_LOGS = {}     # job_id -> list[str]
JOB_LOCKS = {}    # job_id -> threading.Lock()  (guards JOB_LOGS per job)

def safe_append_log(job_id: str, msg: str, replace_last: bool = False):
    """
    Append a log line safely. If replace_last is True, overwrite the last line
    (used for carriage-return / progress updates).
    """
    if job_id not in JOB_LOGS:
        return
    lock = JOB_LOCKS[job_id]
    with lock:
        if replace_last and JOB_LOGS[job_id]:
            JOB_LOGS[job_id][-1] = msg
        else:
            JOB_LOGS[job_id].append(msg)
    # Also print to server console for convenience
    print(msg)


# -----------------------------
# LOG HANDLER (logging module)
# -----------------------------
class YOLOLogHandler(logging.Handler):
    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id

    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            msg = str(record)
        # Logging messages are usually full lines: append normally
        safe_append_log(self.job_id, msg, replace_last=False)


# -----------------------------
# STDOUT / STDERR CAPTURE
# -----------------------------
class StdStreamCapture(io.TextIOBase):
    """
    Redirector for sys.stdout / sys.stderr that captures writes.
    Handles carriage returns (\r) by replacing the last line rather than appending.
    Also passes through to an original stream so console still shows output.
    """
    def __init__(self, job_id: str, orig_stream):
        self.job_id = job_id
        self.orig = orig_stream
        # buffer partial content until newline or flush
        self._buff = ""

    def write(self, s):
        if s is None or s == "":
            return 0
        # Write-through to original stream so you still see console output
        try:
            self.orig.write(s)
        except Exception:
            pass

        # accumulate and split by newline, but preserve carriage-return behavior
        text = s
        # If text contains '\r' without '\n' (progress update), treat as replace_last
        # If text contains '\n', each line will be appended normally.
        parts = text.split('\n')
        for i, part in enumerate(parts):
            if i == 0:
                segment = self._buff + part
            else:
                # previous buffer ended with newline -> append that buffered line
                safe_append_log(self.job_id, self._buff, replace_last=False)
                segment = part
            # If segment contains carriage returns, take the last chunk after the last '\r'
            if '\r' in segment:
                # '\r' indicates overwrite of current progress line (like tqdm)
                # use the portion AFTER the last '\r' as the current progress content
                last = segment.split('\r')[-1]
                # replace last log line with this progress text
                safe_append_log(self.job_id, last, replace_last=True)
                self._buff = ""  # progress handled
            else:
                # if this is the last chunk and original text didn't contain '\n' at end,
                # we should buffer it for the next write
                if i == len(parts) - 1:
                    self._buff = segment
                else:
                    # this chunk ended because of a newline; append it
                    safe_append_log(self.job_id, segment, replace_last=False)
                    self._buff = ""
        return len(s)

    def flush(self):
        # flush any buffered content as a final line
        if self._buff:
            safe_append_log(self.job_id, self._buff, replace_last=False)
            self._buff = ""
        try:
            self.orig.flush()
        except Exception:
            pass


# -----------------------------
# TRAINING FUNCTION (threaded)
# -----------------------------
def run_training(job_id: str, params: dict):
    """
    Runs YOLO training while capturing logs from logging module and stdout/stderr.
    """
    # Prepare
    JOB_STATUS[job_id] = {"status": "running"}
    JOB_LOGS[job_id] = []
    JOB_LOCKS[job_id] = threading.Lock()

    safe_append_log(job_id, f"[{job_id}] Training started")

    # Attach logging handler
    yolo_handler = YOLOLogHandler(job_id)
    yolo_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(yolo_handler)

    # Capture stdout/stderr
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = StdStreamCapture(job_id, orig_stdout)
    sys.stderr = StdStreamCapture(job_id, orig_stderr)

    try:
        # Load model
        model = YOLO(params["model_name"])

        # Normalize device for YOLO (int -> str)
        device = params.get("device")
        if isinstance(device, int):
            device = str(device)

        # Run training (this will produce lots of stdout/stderr and logging)
        results = model.train(
            data=params["data_yaml"],
            epochs=params.get("epochs", 50),
            imgsz=params.get("imgsz", 640),
            device=device,
            batch=params.get("batch", 16),
            lr0=params.get("lr", 0.001),
            **params.get("extra_params", {})
        )

        # Extract save_dir safely
        save_dir = None
        try:
            save_dir = str(results.save_dir)
        except Exception:
            save_dir = None

        JOB_STATUS[job_id] = {"status": "completed", "save_dir": save_dir}
        safe_append_log(job_id, f"[{job_id}] Training completed. save_dir={save_dir}")

    except Exception as e:
        JOB_STATUS[job_id] = {"status": "error", "error": str(e)}
        safe_append_log(job_id, f"[{job_id}] ERROR: {e}")

    finally:
        # Restore stdout/stderr and remove log handler
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass

        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

        try:
            root_logger.removeHandler(yolo_handler)
        except Exception:
            pass

        safe_append_log(job_id, f"[{job_id}] Training finished (handler removed)")


def start_training_job(params: dict) -> str:
    """
    Public helper to spawn a threaded training job.
    Returns job_id immediately.
    """
    job_id = str(uuid.uuid4())
    # initialize structures so websocket clients can connect immediately
    JOB_STATUS[job_id] = {"status": "queued"}
    JOB_LOGS[job_id] = []
    JOB_LOCKS[job_id] = threading.Lock()

    thread = threading.Thread(target=run_training, args=(job_id, params), daemon=True)
    thread.start()
    return job_id


# -----------------------------
# FASTAPI APP + WS
# -----------------------------
app = FastAPI()


class TrainRequest(BaseModel):
    model_name: str
    data_yaml: str
    epochs: int = 50
    imgsz: int = 640
    device: object = "0"   # accepts int, "0", or [0,1]
    batch: int = 16
    lr: float = 0.001
    extra_params: dict = {}  # pass other YOLO train params if needed


@app.post("/train")
def start_training_endpoint(body: TrainRequest):
    params = body.dict()
    # ensure device normalized (int -> str) is handled in run_training
    job_id = start_training_job(params)
    return {"job_id": job_id, "status": "started"}


@app.get("/train/status/{job_id}")
def get_status(job_id: str):
    return JOB_STATUS.get(job_id, {"error": "invalid job_id"})


@app.websocket("/ws/train/{job_id}")
async def ws_train_logs(websocket: WebSocket, job_id: str):
    await websocket.accept()

    if job_id not in JOB_LOGS:
        await websocket.send_text("Invalid job_id")
        await websocket.close()
        return

    last = 0
    try:
        # continuously stream logs as they come
        while True:
            # copy under lock to avoid race conditions
            with JOB_LOCKS[job_id]:
                logs_snapshot = JOB_LOGS[job_id][:]
            if last < len(logs_snapshot):
                for line in logs_snapshot[last:]:
                    await websocket.send_text(line)
                last = len(logs_snapshot)
            # stop if job finished and there are no new logs for a short while
            status = JOB_STATUS.get(job_id, {}).get("status")
            if status in ("completed", "error") and last >= len(logs_snapshot):
                # send a final notice and close
                await websocket.send_text(f"[{job_id}] stream_closed status={status}")
                await websocket.close()
                return
            await asyncio.sleep(0.15)
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
