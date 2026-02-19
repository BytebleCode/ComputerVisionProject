"""
train.py - Train a YOLOv6 model for a named project.

Usage:
  python train.py --project cars --create                    # create new project
  python train.py --project cars --import /path/to/roboflow  # import Roboflow export
  python train.py --project cars                             # train the project
  python train.py --project cars --resume                    # resume interrupted training
  python train.py --list                                     # list all projects
"""

import argparse
import os
import shutil
import subprocess
import sys
import yaml
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT, "venv")
PROJECTS_DIR = os.path.join(ROOT, "projects")
YOLOV6_DIR = os.path.join(ROOT, "YOLOv6")
WEIGHTS_DIR = os.path.join(ROOT, "weights")


def get_venv_python():
    if sys.platform == "win32":
        venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(VENV_DIR, "bin", "python")
    if os.path.isfile(venv_python):
        return venv_python
    return get_venv_python()

WEIGHT_FILES = {
    "n": "yolov6n.pt",
    "s": "yolov6s.pt",
    "m": "yolov6m.pt",
    "l": "yolov6l.pt",
    "lite_s": "yolov6lite_s.pt",
    "lite_m": "yolov6lite_m.pt",
    "lite_l": "yolov6lite_l.pt",
}

FINETUNE_CONFIGS = {
    "n": "configs/yolov6n_finetune.py",
    "s": "configs/yolov6s_finetune.py",
    "m": "configs/yolov6m_finetune.py",
    "l": "configs/yolov6l_finetune.py",
    "lite_s": "configs/yolov6_lite/yolov6_lite_s_finetune.py",
    "lite_m": "configs/yolov6_lite/yolov6_lite_m_finetune.py",
    "lite_l": "configs/yolov6_lite/yolov6_lite_l_finetune.py",
}

DEFAULT_CONFIG = {
    "class_names": ["object"],
    "model_size": "s",
    "batch_size": 16,
    "epochs": 100,
    "img_size": 640,
    "device": "0",
    "save_graphs": True,
    "show_graphs": True,
}


def get_project_dir(name):
    return os.path.join(PROJECTS_DIR, name)


def create_project(name):
    project_dir = get_project_dir(name)
    if os.path.isdir(project_dir):
        print(f"[ERROR] Project '{name}' already exists at {project_dir}")
        sys.exit(1)

    # Create folder structure
    for sub in [
        "datasets/images/train",
        "datasets/images/val",
        "datasets/labels/train",
        "datasets/labels/val",
        "runs",
    ]:
        os.makedirs(os.path.join(project_dir, sub), exist_ok=True)

    # Write template config
    config = dict(DEFAULT_CONFIG)
    config_path = os.path.join(project_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(f"# Project: {name}\n")
        f.write("# Edit class_names to match your dataset labels (order = class IDs)\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[OK] Project '{name}' created at {project_dir}")
    print()
    print("Next steps:")
    print(f"  1. Put training images in:  projects/{name}/datasets/images/train/")
    print(f"  2. Put training labels in:  projects/{name}/datasets/labels/train/")
    print(f"  3. Put val images in:       projects/{name}/datasets/images/val/")
    print(f"  4. Put val labels in:       projects/{name}/datasets/labels/val/")
    print(f"  5. Edit projects/{name}/config.yaml  (set your class_names)")
    print(f"  6. Run:  python train.py --project {name}")


def import_dataset(name, source_path):
    """Import a Roboflow (or similar) export into a project.

    Handles:
      - 'valid' vs 'val' folder naming
      - Reads data.yaml for class names
      - Copies images and labels into the project structure
    """
    source_path = os.path.abspath(source_path)
    if not os.path.isdir(source_path):
        print(f"[ERROR] Source path not found: {source_path}")
        sys.exit(1)

    project_dir = get_project_dir(name)

    # Create project if it doesn't exist
    if not os.path.isdir(project_dir):
        for sub in [
            "datasets/images/train",
            "datasets/images/val",
            "datasets/labels/train",
            "datasets/labels/val",
            "runs",
        ]:
            os.makedirs(os.path.join(project_dir, sub), exist_ok=True)

    # Detect folder layout
    src_images = os.path.join(source_path, "images")
    src_labels = os.path.join(source_path, "labels")

    # Map source split names to our split names
    # Roboflow uses "valid", we use "val"
    split_map = {"train": "train", "val": "val", "valid": "val", "test": "test"}

    def copy_files(src_dir, dst_dir):
        """Copy all files from src to dst, return count."""
        os.makedirs(dst_dir, exist_ok=True)
        count = 0
        if os.path.isdir(src_dir):
            for f in os.listdir(src_dir):
                src_file = os.path.join(src_dir, f)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, os.path.join(dst_dir, f))
                    count += 1
        return count

    print(f"[IMPORT] Importing dataset into project '{name}'...")
    print(f"  Source: {source_path}")
    print()

    total_images = 0
    total_labels = 0

    for src_split, dst_split in split_map.items():
        src_img_split = os.path.join(src_images, src_split)
        src_lbl_split = os.path.join(src_labels, src_split)

        if not os.path.isdir(src_img_split):
            continue

        dst_img_split = os.path.join(project_dir, "datasets", "images", dst_split)
        dst_lbl_split = os.path.join(project_dir, "datasets", "labels", dst_split)

        img_count = copy_files(src_img_split, dst_img_split)
        lbl_count = copy_files(src_lbl_split, dst_lbl_split)

        total_images += img_count
        total_labels += lbl_count
        print(f"  {src_split:>6} -> {dst_split:<5}  {img_count} images, {lbl_count} labels")

    # Read class names from source data.yaml if it exists
    src_data_yaml = os.path.join(source_path, "data.yaml")
    class_names = DEFAULT_CONFIG["class_names"]
    if os.path.isfile(src_data_yaml):
        with open(src_data_yaml) as f:
            src_data = yaml.safe_load(f)
        if "names" in src_data:
            class_names = src_data["names"]
            print(f"\n  Classes found: {class_names}")

    # Write project config
    config = dict(DEFAULT_CONFIG)
    config["class_names"] = class_names
    config_path = os.path.join(project_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(f"# Project: {name}\n")
        f.write(f"# Imported from: {source_path}\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print()
    print(f"[OK] Imported {total_images} images and {total_labels} labels.")
    print(f"  Config: {config_path}")
    print(f"  Run:    python train.py --project {name}")


def list_projects():
    if not os.path.isdir(PROJECTS_DIR):
        print("No projects found. Run: python train.py --project <name> --create")
        return

    projects = sorted(
        d for d in os.listdir(PROJECTS_DIR)
        if os.path.isdir(os.path.join(PROJECTS_DIR, d))
    )
    if not projects:
        print("No projects found. Run: python train.py --project <name> --create")
        return

    print(f"{'PROJECT':<20} {'STATUS':<12} {'BEST mAP':<12} {'EPOCHS':<10}")
    print("-" * 54)

    for name in projects:
        project_dir = get_project_dir(name)
        best_ckpt = find_best_checkpoint(project_dir)
        # Check for training results CSV to get mAP
        status = "trained" if best_ckpt else "untrained"
        map_str = "-"
        epochs_str = "-"

        results_csv = find_results_csv(project_dir)
        if results_csv:
            import pandas as pd
            try:
                df = pd.read_csv(results_csv)
                # Try common column names
                for col in df.columns:
                    if "mAP" in col and "0.5" in col and "0.95" not in col:
                        map_str = f"{df[col].max():.3f}"
                        break
                epochs_str = str(len(df))
            except Exception:
                pass

        print(f"{name:<20} {status:<12} {map_str:<12} {epochs_str:<10}")


def find_best_checkpoint(project_dir):
    runs_dir = os.path.join(project_dir, "runs")
    # Look for best_ckpt.pt in any run folder
    matches = glob.glob(os.path.join(runs_dir, "**", "best_ckpt.pt"), recursive=True)
    if not matches:
        return None
    # Return most recently modified
    return max(matches, key=os.path.getmtime)


def find_results_csv(project_dir):
    runs_dir = os.path.join(project_dir, "runs")
    matches = glob.glob(os.path.join(runs_dir, "**", "results.csv"), recursive=True)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_config(project_dir):
    config_path = os.path.join(project_dir, "config.yaml")
    if not os.path.isfile(config_path):
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_data_yaml(project_dir, config):
    """Generate the YOLOv6-format data YAML for this project."""
    datasets_dir = os.path.join(project_dir, "datasets")
    data_yaml_path = os.path.join(project_dir, "data.yaml")

    data = {
        "train": os.path.join(datasets_dir, "images", "train"),
        "val": os.path.join(datasets_dir, "images", "val"),
        "is_coco": False,
        "nc": len(config["class_names"]),
        "names": config["class_names"],
    }

    with open(data_yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return data_yaml_path


def validate_dataset(project_dir):
    """Check that the dataset has images and labels."""
    train_imgs = os.path.join(project_dir, "datasets", "images", "train")
    val_imgs = os.path.join(project_dir, "datasets", "images", "val")
    train_labels = os.path.join(project_dir, "datasets", "labels", "train")

    train_img_count = len([
        f for f in os.listdir(train_imgs)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]) if os.path.isdir(train_imgs) else 0

    val_img_count = len([
        f for f in os.listdir(val_imgs)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]) if os.path.isdir(val_imgs) else 0

    train_label_count = len([
        f for f in os.listdir(train_labels)
        if f.endswith(".txt")
    ]) if os.path.isdir(train_labels) else 0

    if train_img_count == 0:
        print(f"[ERROR] No training images found in {train_imgs}")
        sys.exit(1)
    if val_img_count == 0:
        print(f"[ERROR] No validation images found in {val_imgs}")
        sys.exit(1)
    if train_label_count == 0:
        print(f"[ERROR] No training labels found in {train_labels}")
        sys.exit(1)

    print(f"  Train: {train_img_count} images, {train_label_count} labels")
    print(f"  Val:   {val_img_count} images")


def train(name, resume=False):
    project_dir = get_project_dir(name)
    if not os.path.isdir(project_dir):
        print(f"[ERROR] Project '{name}' not found. Create it first:")
        print(f"  python train.py --project {name} --create")
        sys.exit(1)

    if not os.path.isdir(YOLOV6_DIR):
        print("[ERROR] YOLOv6 not found. Run setup.py first:")
        print("  python setup.py")
        sys.exit(1)

    config = load_config(project_dir)
    model_size = config.get("model_size", "s")
    batch_size = config.get("batch_size", 16)
    epochs = config.get("epochs", 100)
    img_size = config.get("img_size", 640)
    device = str(config.get("device", "0"))

    print(f"[TRAIN] Project: {name}")
    print(f"  Model: YOLOv6-{model_size}")
    print(f"  Classes: {config['class_names']}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, Img size: {img_size}")
    print(f"  Device: {device}")
    print()

    # Validate dataset
    print("Checking dataset...")
    validate_dataset(project_dir)
    print()

    # Generate data YAML
    data_yaml = generate_data_yaml(project_dir, config)

    # Resolve finetune config and weights
    base_conf = os.path.join(YOLOV6_DIR, FINETUNE_CONFIGS.get(model_size, FINETUNE_CONFIGS["s"]))
    pretrained = os.path.join(WEIGHTS_DIR, WEIGHT_FILES.get(model_size, WEIGHT_FILES["s"]))

    # Generate a project-specific config with the correct pretrained weights path
    project_conf = os.path.join(project_dir, "finetune_config.py")
    with open(base_conf) as f:
        conf_text = f.read()
    # Replace the pretrained path with our absolute path
    conf_text = conf_text.replace(
        "pretrained='weights/",
        f"pretrained='{pretrained.replace(os.sep, '/')}".rstrip("/") + "/../weights/"
    )
    # Simpler: just replace the whole pretrained line
    import re
    conf_text = re.sub(
        r"pretrained='[^']*'",
        f"pretrained='{pretrained.replace(os.sep, '/')}'",
        conf_text
    )
    with open(project_conf, "w") as f:
        f.write(conf_text)

    output_dir = os.path.join(project_dir, "runs")

    # Build command (use actual YOLOv6 arg names from its argparser)
    cmd = [
        get_venv_python(),
        os.path.join(YOLOV6_DIR, "tools", "train.py"),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--img-size", str(img_size),
        "--conf-file", project_conf,
        "--data-path", data_yaml,
        "--fuse_ab",
        "--device", device,
        "--output-dir", output_dir,
    ]

    if resume:
        last_ckpt = find_best_checkpoint(project_dir)
        if last_ckpt:
            ckpt_dir = os.path.dirname(last_ckpt)
            last = os.path.join(ckpt_dir, "last_ckpt.pt")
            if os.path.isfile(last):
                cmd.extend(["--resume", last])
                print(f"Resuming from: {last}")
            else:
                print("[WARN] No last_ckpt.pt found, starting fresh.")
        else:
            print("[WARN] No checkpoint found, starting fresh.")

    print("Starting training...")
    print(f"  Command: {' '.join(cmd)}")
    print()

    # YOLOv6 must be run from its own directory so it can find the yolov6 module
    env = os.environ.copy()
    env["PYTHONPATH"] = YOLOV6_DIR + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call(cmd, cwd=YOLOV6_DIR, env=env)

    # Find best checkpoint
    best = find_best_checkpoint(project_dir)
    print()
    print("=" * 60)
    print("  Training complete!")
    print("=" * 60)
    if best:
        print(f"  Best checkpoint: {best}")

    # Auto-generate graphs
    if config.get("save_graphs", True):
        print()
        print("Generating training graphs...")
        graphs_cmd = [
            get_venv_python(),
            os.path.join(ROOT, "graphs.py"),
            "--project", name,
        ]
        if not config.get("show_graphs", True):
            graphs_cmd.append("--save-only")
        try:
            subprocess.check_call(graphs_cmd)
        except Exception as e:
            print(f"[WARN] Could not generate graphs: {e}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv6 Project Trainer")
    parser.add_argument("--project", type=str, help="Project name")
    parser.add_argument("--create", action="store_true", help="Create a new project")
    parser.add_argument("--import", dest="import_path", type=str,
                        help="Import a Roboflow/YOLO dataset export folder")
    parser.add_argument("--list", action="store_true", help="List all projects")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()

    if args.list:
        list_projects()
        return

    if not args.project:
        parser.print_help()
        print("\nExamples:")
        print("  python train.py --project cars --create")
        print('  python train.py --project cars --import "/path/to/roboflow/export"')
        print("  python train.py --project cars")
        print("  python train.py --list")
        return

    if args.import_path:
        import_dataset(args.project, args.import_path)
    elif args.create:
        create_project(args.project)
    else:
        train(args.project, resume=args.resume)


if __name__ == "__main__":
    main()
