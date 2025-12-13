
from ultralytics import YOLO

# ---------- YOLO Train Function ----------
def train_yolo_model(config):
    print(f"\n🚀 Starting YOLO Training with Config:\n{config}\n")

    # Load model
    model = YOLO(config.model_name)

    # Train
    results = model.train(
        data=config.data_yaml,
        epochs=config.epochs,
        imgsz=config.imgsz,
        device=config.device,
        batch=config.batch,
        lr0=config.lr
    )

    return results