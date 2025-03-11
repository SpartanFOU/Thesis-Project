import os
import subprocess
import sys
from pathlib import Path

def create_venv(venv_path='venv'):
    if not Path(venv_path).exists():
        print(f"[INFO] Creating virtual environment at '{venv_path}'...")
        subprocess.run([sys.executable, "-m", "venv", venv_path])
    else:
        print("[INFO] Virtual environment already exists.")

def install_requirements(venv_path='venv'):
    pip = Path(venv_path) / ('Scripts' if os.name == 'nt' else 'bin') / 'pip'
    print("[INFO] Installing requirements...")
    subprocess.run([str(pip), "install", "-r", "requirements.txt"])

def install_project_package(venv_path='venv'):
    pip = Path(venv_path) / ('Scripts' if os.name == 'nt' else 'bin') / 'pip'
    print("[INFO] Installing project package (editable)...")
    subprocess.run([str(pip), "install", "-e", "."])

if __name__ == "__main__":
    create_venv()
    install_requirements()
    install_project_package()
    print("\n✅ Environment is ready. Activate the venv and run your notebook.")
