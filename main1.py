from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import zipfile
import os
import shutil
import random
import yaml
import uuid

app = FastAPI()
TRAIN_RATIO = 0.8
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ------------------ UTILS ---------------------

def parse_classes(classes: str):
    classes = classes.strip()

    # If user provides list format → ["car","person"]
    if classes.startswith("[") and classes.endswith("]"):
        fixed = classes[1:-1].split(",")
        return [c.strip().strip("'\"") for c in fixed]

    # If comma separated → car,person
    if "," in classes:
        return [c.strip() for c in classes.split(",") if c.strip()]

    # Single class → "car"
    return [classes]


def unzip_and_locate(zip_file, dest):
    """Unzip and detect folder containing images/labels."""
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(dest)

    for root, dirs, files in os.walk(dest):
        if any(f.lower().endswith(IMAGE_EXT) for f in files) or any(f.endswith(".txt") for f in files):
            return root
    return dest


def create_yaml(output_dir, class_list):
    yaml_data = {
        "train": "../train/images",
        "val": "../valid/images",
        "nc": len(class_list),
        "names": class_list
    }

    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)


# ---------------- API -------------------

@app.post("/prepare-dataset")
async def prepare_dataset(
    images_zip: UploadFile,
    labels_zip: UploadFile,
    classes: str = Form(...)
):
    try:
        class_list = parse_classes(classes)

        job_id = str(uuid.uuid4())
        base_dir = f"dataset_{job_id}"
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
            "classes": class_list
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
