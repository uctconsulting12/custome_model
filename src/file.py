import zipfile
import os

import yaml


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