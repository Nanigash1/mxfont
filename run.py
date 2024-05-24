import subprocess

def install_dependencies():
    """Installs the required Python packages using pip."""
    packages = ["sconf", "numpy", "scipy", "scikit-image", "tqdm", "jsonlib-python3", "fonttools"]
    subprocess.run(["pip", "install"] + packages)

def run_training():
    """Executes the mxfont training script."""
    subprocess.run(["python", "mxfont/train.py", "mxfont/cfgs/train.yaml"])

if __name__ == "__main__":
    install_dependencies()
    run_training()