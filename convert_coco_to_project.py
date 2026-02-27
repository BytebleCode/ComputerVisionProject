"""
One-time script to convert COCO-format traffic light annotations to a YOLO project.

Source: pullimgfolder/Traffic_Lights/train/_annotations.coco.json
Target: projects/traffic_light_detector/
"""

import json
import os
import random
import shutil
from pathlib import Path

# --- Config ---
COCO_JSON = "pullimgfolder/Traffic_Lights/train/_annotations.coco.json"
IMAGE_DIR = "pullimgfolder/Traffic_Lights/train"
PROJECT_DIR = "projects/traffic_light_detector"
SPLIT_RATIO = 0.8  # 80% train, 20% val
SEED = 42

# COCO category_id -> YOLO class_id
CATEGORY_MAP = {1: 0, 2: 1, 3: 2}  # green=0, red=1, yellow=2
CLASS_NAMES = ["green", "red", "yellow"]


def main():
    root = Path(__file__).resolve().parent
    coco_path = root / COCO_JSON
    img_dir = root / IMAGE_DIR
    proj_dir = root / PROJECT_DIR

    # Load COCO annotations
    with open(coco_path, "r") as f:
        coco = json.load(f)

    # Build lookup: image_id -> image info
    images = {img["id"]: img for img in coco["images"]}

    # Build lookup: image_id -> list of annotations
    annotations = {}
    skipped = 0
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in CATEGORY_MAP:
            skipped += 1
            continue
        annotations.setdefault(ann["image_id"], []).append(ann)
    if skipped:
        print(f"Skipped {skipped} annotations with unmapped category IDs")

    # Split images 80/20
    image_ids = sorted(images.keys())
    random.seed(SEED)
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * SPLIT_RATIO)
    splits = {
        "train": image_ids[:split_idx],
        "val": image_ids[split_idx:],
    }

    # Create directory structure
    for split in ("train", "val"):
        (proj_dir / "datasets" / "images" / split).mkdir(parents=True, exist_ok=True)
        (proj_dir / "datasets" / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Process each split
    for split, ids in splits.items():
        for img_id in ids:
            img_info = images[img_id]
            fname = img_info["file_name"]
            w, h = img_info["width"], img_info["height"]

            # Copy image
            src = img_dir / fname
            dst = proj_dir / "datasets" / "images" / split / fname
            shutil.copy2(src, dst)

            # Write YOLO label file
            stem = Path(fname).stem
            label_path = proj_dir / "datasets" / "labels" / split / f"{stem}.txt"
            anns = annotations.get(img_id, [])
            with open(label_path, "w") as lf:
                for ann in anns:
                    cls_id = CATEGORY_MAP[ann["category_id"]]
                    bx, by, bw, bh = ann["bbox"]  # COCO: top-left x,y,w,h (pixels)
                    # Convert to YOLO: center_x, center_y, width, height (normalized)
                    cx = (bx + bw / 2) / w
                    cy = (by + bh / 2) / h
                    nw = bw / w
                    nh = bh / h
                    lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        print(f"{split}: {len(ids)} images")

    # Write config.yaml
    config_path = proj_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write("# Project: traffic_light_detector\n")
        f.write("# Converted from COCO format\n\n")
        f.write("class_names:\n")
        for name in CLASS_NAMES:
            f.write(f"- {name}\n")
        f.write("model_size: s\n")
        f.write("batch_size: 16\n")
        f.write("epochs: 25\n")
        f.write("img_size: 416\n")
        f.write("device: '0'\n")
        f.write("save_graphs: true\n")
        f.write("show_graphs: true\n")

    print(f"\nProject created at: {proj_dir}")
    print(f"Config: {config_path}")


if __name__ == "__main__":
    main()
