"""
detect.py - Run inference with a trained YOLOv6 model.

Usage:
  python detect.py --project cars --source image.jpg
  python detect.py --project cars --source ./my_images/
  python detect.py --project cars --source video.mp4
"""

import argparse
import os
import subprocess
import sys
import yaml
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT, "venv")
PROJECTS_DIR = os.path.join(ROOT, "projects")
YOLOV6_DIR = os.path.join(ROOT, "YOLOv6")


def get_venv_python():
    if sys.platform == "win32":
        venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(VENV_DIR, "bin", "python")
    if os.path.isfile(venv_python):
        return venv_python
    return get_venv_python()


def find_best_checkpoint(project_dir):
    runs_dir = os.path.join(project_dir, "runs")
    matches = glob.glob(os.path.join(runs_dir, "**", "best_ckpt.pt"), recursive=True)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv6 inference")
    parser.add_argument("--project", type=str, required=True, help="Project name")
    parser.add_argument("--source", type=str, required=True,
                        help="Image file, folder, or video to run detection on")
    parser.add_argument("--conf-thresh", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou-thresh", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")
    args = parser.parse_args()

    project_dir = os.path.join(PROJECTS_DIR, args.project)
    if not os.path.isdir(project_dir):
        print(f"[ERROR] Project '{args.project}' not found.")
        sys.exit(1)

    # Load config
    config_path = os.path.join(project_dir, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = str(config.get("device", "0"))
    img_size = config.get("img_size", 640)

    best_ckpt = find_best_checkpoint(project_dir)
    if not best_ckpt:
        print("[ERROR] No trained checkpoint found. Train the model first:")
        print(f"  python train.py --project {args.project}")
        sys.exit(1)

    output_dir = os.path.join(project_dir, "runs", "detect")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[DETECT] Project: {args.project}")
    print(f"  Checkpoint: {best_ckpt}")
    print(f"  Source: {args.source}")
    print(f"  Device: {device}")
    print()

    cmd = [
        get_venv_python(),
        os.path.join(YOLOV6_DIR, "tools", "infer.py"),
        "--weights", best_ckpt,
        "--source", args.source,
        "--img-size", str(img_size),
        "--conf-thres", str(args.conf_thresh),
        "--iou-thres", str(args.iou_thresh),
        "--device", device,
        "--save-dir", output_dir,
    ]

    print(f"  Command: {' '.join(cmd)}")
    print()
    env = os.environ.copy()
    env["PYTHONPATH"] = YOLOV6_DIR + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call(cmd, cwd=YOLOV6_DIR, env=env)

    print()
    print(f"[OK] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
