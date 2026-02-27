# FineTuneYOLO

A framework for fine-tuning YOLOv6 models on custom datasets. Each detector is its own "project" with separate config, data, and weights.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python setup.py
```

This clones YOLOv6, installs dependencies, and downloads pre-trained weights.

## Quick Start

```bash
python train.py --project my_detector --create
# Add images to projects/my_detector/datasets/images/train/ and val/
# Add labels to projects/my_detector/datasets/labels/train/ and val/
# Edit projects/my_detector/config.yaml with your class names

python train.py --project my_detector
python evaluate.py --project my_detector
python detect.py --project my_detector --source path/to/image.jpg
```

## Config

Each project has a config.yaml:

```yaml
class_names:
  - cat
  - dog
model_size: "s"       # n, s, m, l, lite_s, lite_m, lite_l
batch_size: 16
epochs: 100
img_size: 640
device: "0"           # GPU index or "cpu"
```

## Models

- n (Nano) -fastest, lowest accuracy
- s (Small) -good balance (default)
- m (Medium) -higher accuracy
- l (Large) -best accuracy
- lite_s, lite_m, lite_l -lightweight variants for mobile/embedded

## Dataset Format

Each image needs a matching .txt label file. One line per object:

```
<class_id> <center_x> <center_y> <width> <height>
```

All coordinates are normalized 0-1. Class IDs are zero-based, matching the order in class_names.

## Commands

```bash
python train.py --project <name> --create          # Create a new project
python train.py --project <name>                    # Train
python train.py --project <name> --resume           # Resume training
python train.py --list                              # List all projects
python train.py --project <name> --import <path>    # Import a Roboflow/YOLO dataset
python evaluate.py --project <name>                 # Evaluate (mAP metrics)
python detect.py --project <name> --source <path>   # Run inference
python graphs.py --project <name>                   # Plot training curves
python graphs.py --project <name> --compare <name2> # Compare two projects
python convert_coco_to_project.py                   # Convert COCO annotations to YOLO format
```

## COCO to YOLO Conversion

If your dataset uses COCO format (a single JSON file with all annotations), use convert_coco_to_project.py to convert it into a YOLO project.

COCO format stores bounding boxes as pixel values in top-left [x, y, width, height] format. YOLO format needs them as normalized center coordinates [center_x, center_y, width, height] where all values are 0-1 relative to the image size. The script handles this conversion automatically.

To use it, edit the variables at the top of the script:

```python
COCO_JSON = "path/to/your/_annotations.coco.json"
IMAGE_DIR = "path/to/your/images"
PROJECT_DIR = "projects/your_project_name"

CATEGORY_MAP = {1: 0, 2: 1, 3: 2}       # COCO category ID -> YOLO class ID
CLASS_NAMES = ["class_a", "class_b", "class_c"]
```

CATEGORY_MAP maps COCO category IDs (which can be any number) to sequential YOLO class IDs starting from 0. The order must match CLASS_NAMES.

Then run:

```bash
python convert_coco_to_project.py
```

The script will:

1. Load the COCO JSON and parse all annotations
2. Split images 80/20 into train/val sets (seeded for reproducibility)
3. Copy images into the project's datasets/images/train/ and val/ folders
4. Convert each bounding box from COCO pixel format to YOLO normalized format and write .txt label files
5. Generate a config.yaml for the project with your class names

After conversion, the project is ready to train with python train.py --project your_project_name.

## Example Projects

- **color_sorter** -9 classes, 953 images -shape and color detection
- **lane_assist** -12 classes, 329 images -lane marking detection
- **traffic_light_detector** -3 classes, 2575 images -traffic light detection

## Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA GPU recommended
