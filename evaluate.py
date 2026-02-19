"""
evaluate.py - Evaluate a trained YOLOv6 model for a project.

Usage:
  python evaluate.py --project cars
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
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv6 project")
    parser.add_argument("--project", type=str, required=True, help="Project name")
    parser.add_argument("--task", type=str, default="val", choices=["val", "test"],
                        help="Evaluation split (default: val)")
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

    # Find data yaml and best checkpoint
    data_yaml = os.path.join(project_dir, "data.yaml")
    if not os.path.isfile(data_yaml):
        print("[ERROR] data.yaml not found. Train the model first:")
        print(f"  python train.py --project {args.project}")
        sys.exit(1)

    best_ckpt = find_best_checkpoint(project_dir)
    if not best_ckpt:
        print("[ERROR] No trained checkpoint found. Train the model first:")
        print(f"  python train.py --project {args.project}")
        sys.exit(1)

    print(f"[EVAL] Project: {args.project}")
    print(f"  Checkpoint: {best_ckpt}")
    print(f"  Task: {args.task}")
    print(f"  Device: {device}")
    print()

    cmd = [
        get_venv_python(),
        os.path.join(YOLOV6_DIR, "tools", "eval.py"),
        "--data", data_yaml,
        "--weights", best_ckpt,
        "--task", args.task,
        "--img-size", str(img_size),
        "--device", device,
    ]

    print(f"  Command: {' '.join(cmd)}")
    print()
    env = os.environ.copy()
    env["PYTHONPATH"] = YOLOV6_DIR + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call(cmd, cwd=YOLOV6_DIR, env=env)


if __name__ == "__main__":
    main()
