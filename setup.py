"""
setup.py - One-time setup for YOLOv6 fine-tuning environment.

Clones the YOLOv6 repo, installs dependencies, downloads pretrained weights,
and creates the projects folder structure.
"""

import os
import subprocess
import sys
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT, "venv")
YOLOV6_DIR = os.path.join(ROOT, "YOLOv6")
WEIGHTS_DIR = os.path.join(ROOT, "weights")
PROJECTS_DIR = os.path.join(ROOT, "projects")


def get_venv_python():
    """Return the venv Python path, or sys.executable as fallback."""
    if sys.platform == "win32":
        venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(VENV_DIR, "bin", "python")
    if os.path.isfile(venv_python):
        return venv_python
    return sys.executable

WEIGHT_URLS = {
    "n": "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt",
    "s": "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt",
    "m": "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m.pt",
    "l": "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l.pt",
    "lite_s": "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_s.pt",
    "lite_m": "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_m.pt",
    "lite_l": "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_l.pt",
}


def clone_yolov6():
    if os.path.isdir(YOLOV6_DIR):
        print("[OK] YOLOv6 repo already exists, skipping clone.")
        return
    print("[1/4] Cloning YOLOv6 repository...")
    subprocess.check_call(
        ["git", "clone", "https://github.com/meituan/YOLOv6.git", YOLOV6_DIR]
    )
    print("[OK] YOLOv6 cloned.")


def install_dependencies():
    python = get_venv_python()
    print(f"[2/4] Installing dependencies (using {python})...")
    subprocess.check_call(
        [python, "-m", "pip", "install", "-r", os.path.join(ROOT, "requirements.txt")]
    )
    yolov6_req = os.path.join(YOLOV6_DIR, "requirements.txt")
    if os.path.isfile(yolov6_req):
        # Install YOLOv6 deps line by line, skipping optional ones that fail
        # (e.g. onnx-simplifier needs cmake, thop may fail - neither needed for training)
        with open(yolov6_req) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pkg = line.split("#")[0].strip()  # strip inline comments
                if not pkg:
                    continue
                try:
                    subprocess.check_call(
                        [python, "-m", "pip", "install", pkg],
                        stdout=subprocess.DEVNULL,
                    )
                    print(f"  [OK] {pkg}")
                except subprocess.CalledProcessError:
                    print(f"  [SKIP] {pkg} (optional, install failed)")
    print("[OK] Dependencies installed.")


def download_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print("[3/4] Downloading pretrained weights...")
    for size, url in WEIGHT_URLS.items():
        filename = os.path.basename(url)
        dest = os.path.join(WEIGHTS_DIR, filename)
        if os.path.isfile(dest):
            print(f"  [OK] {filename} already exists, skipping.")
            continue
        print(f"  Downloading {filename}...")
        resp = requests.get(url, stream=True, allow_redirects=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  [OK] {filename} saved.")
    print("[OK] All weights downloaded.")


def create_projects_dir():
    print("[4/4] Creating projects directory...")
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    print("[OK] projects/ ready.")


def main():
    print("=" * 60)
    print("  YOLOv6 Fine-Tuning Setup")
    print("=" * 60)
    print()

    clone_yolov6()
    install_dependencies()
    download_weights()
    create_projects_dir()

    print()
    print("=" * 60)
    print("  Setup complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Create a new project:")
    print("       python train.py --project <name> --create")
    print("  2. Add your images and labels to the project datasets/ folder")
    print("  3. Edit the project config.yaml with your class names")
    print("  4. Train:")
    print("       python train.py --project <name>")
    print()


if __name__ == "__main__":
    main()
