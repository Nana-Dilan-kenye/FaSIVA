#!/usr/bin/env python3
"""
Setup script for FaSIVA implementation
Creates virtual environment and installs compatible dependencies
"""
import os
import sys
import subprocess
import platform

from main import FaSIVA

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

def create_virtual_env():
    """Create virtual environment"""
    print("\nCreating virtual environment...")
    
    venv_dir = "fasiva_env"
    
    if os.path.exists(venv_dir):
        print(f"Virtual environment '{venv_dir}' already exists")
        return venv_dir
    
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        print(f"✓ Virtual environment created: {venv_dir}")
        return venv_dir
    except Exception as e:
        print(f"Failed to create virtual environment: {e}")
        print("Creating project directories only...")
        return None

def get_pip_path(venv_dir):
    """Get pip path in virtual environment"""
    if platform.system() == "Windows":
        pip_path = os.path.join(venv_dir, "Scripts", "pip")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")
    
    if os.path.exists(pip_path):
        return pip_path
    elif os.path.exists(pip_path + "3"):
        return pip_path + "3"
    else:
        return "pip"

def get_python_path(venv_dir):
    """Get python path in virtual environment"""
    if platform.system() == "Windows":
        python_path = os.path.join(venv_dir, "Scripts", "python")
    else:
        python_path = os.path.join(venv_dir, "bin", "python")
    
    if os.path.exists(python_path):
        return python_path
    elif os.path.exists(python_path + "3"):
        return python_path + "3"
    else:
        return sys.executable

def install_dependencies(venv_dir):
    """Install compatible dependencies"""
    print("\nInstalling dependencies...")
    
    pip_path = get_pip_path(venv_dir)
    
    # First upgrade pip
    try:
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
    except:
        pass
    
    # Install compatible versions (order matters!)
    packages = [
        "numpy==1.24.3",  # Stable version compatible with scipy
        "scipy==1.10.1",  # Compatible with numpy 1.24.3
        "opencv-python==4.8.1.78",
        "Pillow==10.0.0",
        "tqdm==4.65.0",
        "requests==2.31.0",
        "matplotlib==3.7.1",
        "scikit-learn==1.3.0",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "facenet-pytorch==2.5.3",
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run([pip_path, "install", package], check=True)
            print(f"✓ {package}")
        except Exception as e:
            print(f"✗ Failed to install {package}: {e}")
    
    # Install dlib (may require special handling)
    print("\nInstalling dlib...")
    try:
        # Try standard installation first
        subprocess.run([pip_path, "install", "dlib==19.24.2"], check=True)
        print("✓ dlib")
    except:
        print("Note: dlib installation may fail. Will use alternative face detection.")
    
    print("\n✓ All dependencies installed!")

def create_directories():
    """Create project directories"""
    print("\nCreating project directories...")
    
    directories = [
        "models",
        "datasets",
        "datasets/train",
        "datasets/val", 
        "datasets/test",
        "logs",
        "signatures"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created: {dir_path}")

def download_resources():
    """Download required resources"""
    print("\nDownloading resources...")
    
    import urllib.request
    import bz2
    
    # Download dlib shape predictor
    shape_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    shape_path = "models/shape_predictor_68_face_landmarks.dat"
    
    if not os.path.exists(shape_path):
        print("Downloading facial landmark predictor...")
        try:
            # Download compressed file
            temp_path = shape_path + ".bz2"
            urllib.request.urlretrieve(shape_url, temp_path)
            
            # Decompress
            with open(temp_path, 'rb') as f:
                compressed = f.read()
            
            decompressed = bz2.decompress(compressed)
            
            # Save decompressed file
            with open(shape_path, 'wb') as f:
                f.write(decompressed)
            
            # Clean up
            os.remove(temp_path)
            print(f"✓ Downloaded: {shape_path}")
        except Exception as e:
            print(f"Note: Could not download shape predictor: {e}")
    else:
        print("✓ Shape predictor already exists")

def create_activation_script(venv_dir):
    """Create activation script"""
    print("\nCreating activation script...")
    
    if platform.system() == "Windows":
        script = f"""@echo off
echo Activating FaSIVA environment...
call {venv_dir}\\Scripts\\activate
echo.
echo To run FaSIVA:
echo python main.py
echo.
cmd /k
"""
        script_path = "activate_fasiva.bat"
    else:
        script = f"""#!/bin/bash
echo "Activating FaSIVA environment..."
source {venv_dir}/bin/activate
echo ""
echo "To run FaSIVA:"
echo "python main.py"
echo ""
exec $SHELL
"""
        script_path = "activate_fasiva.sh"
        # Make executable
        os.chmod(script_path, 0o755)
    
    with open(script_path, "w") as f:
        f.write(script)
    
    print(f"✓ Created: {script_path}")

def create_requirements_txt():
    """Create requirements.txt file"""
    print("\nCreating requirements.txt...")
    
    requirements = """# FaSIVA Requirements
# Compatible versions tested together

# Core numerical libraries
numpy==1.24.3
scipy==1.10.1

# Image processing
opencv-python==4.8.1.78
Pillow==10.0.0

# Deep learning
torch==2.0.1
torchvision==0.15.2
facenet-pytorch==2.5.3

# Face detection (install may require system dependencies)
dlib==19.24.2

# Utilities
tqdm==4.65.0
requests==2.31.0
matplotlib==3.7.1
scikit-learn==1.3.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✓ Created: requirements.txt")

def create_simple_demo():
    """Create simple demo script"""
    print("\nCreating demo scripts...")
    
    demo_code = """#!/usr/bin/env python3
    """
    # Simple FaSIVA Demo
    """
    import cv2
    import numpy as np
    import os

    print("FaSIVA - Facial Signature Demo")
    print("=" * 40)

    # Create a test face image
    def create_test_face():
        \"\"\"Create a synthetic face for testing\"\"\"
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Face
        cv2.ellipse(img, (100, 100), (70, 90), 0, 0, 360, (255, 220, 180), -1)
        
        # Eyes
        cv2.circle(img, (70, 80), 15, (0, 0, 0), -1)
        cv2.circle(img, (130, 80), 15, (0, 0, 0), -1)
        
        # Mouth
        cv2.ellipse(img, (100, 130), (30, 15), 0, 0, 180, (0, 0, 0), 3)
        
        # Nose
        cv2.line(img, (100, 100), (100, 110), (0, 0, 0), 2)
        
        return img

    # Test basic OpenCV face detection
    print("\\nTesting basic components...")

    # Create test image
    test_face = create_test_face()
    cv2.imwrite("test_face.jpg", test_face)
    print("✓ Created test face image")

    # Try to load OpenCV face detector
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(test_face, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            print(f"✓ Face detected: {len(faces)} face(s) found")
        else:
            print("✗ No face detected (expected for synthetic image)")
        
    except Exception as e:
        print(f"Note: Face detection test failed: {e}")

    # Check for models
    print("\\nChecking for required models...")
    models_dir = "models"
    if os.path.exists(models_dir):
        model_files = os.listdir(models_dir)
        print(f"Models directory contains: {model_files}")
    else:
        print("Models directory not found - will be created when needed")

    print("\\n" + "=" * 40)
    print("Setup complete! Next steps:")
    print("1. Activate the environment:")
    print("   Windows: .\\\\activate_fasiva.bat")
    print("   Mac/Linux: source activate_fasiva.sh")
    print("\\n2. Run the main system:")
    print("   python main.py")
    print("\\n3. For training:")
    print("   python train.py --component prepare")
    """

    with open("demo.py", "w") as f:
        f.write(demo_code)
    
    # Make executable on Unix-like systems
    if platform.system() != "Windows":
        os.chmod("demo.py", 0o755)
    
    print("✓ Created: demo.py")

def main():
    """Main setup function"""
    print("=" * 60)
    print("FaSIVA - Facial Signature Implementation Setup")
    print("=" * 60)
    
    # Check Python
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Download resources
    download_resources()
    
    # Create virtual environment
    venv_dir = create_virtual_env()
    
    if venv_dir:
        # Install dependencies
        install_dependencies(venv_dir)
        
        # Create activation script
        create_activation_script(venv_dir)
    
    # Create requirements file
    create_requirements_txt()
    
    # Create demo
    create_simple_demo()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    
    if venv_dir:
        print(f"\nVirtual environment: {venv_dir}")
        print("\nTo activate:")
        if platform.system() == "Windows":
            print("  .\\activate_fasiva.bat")
        else:
            print("  source activate_fasiva.sh")
    else:
        print("\nUsing system Python")
    
    print("\nTo run a quick test:")
    print("  python demo.py")
    
    print("\nTo run the full system:")
    print("  python main.py")

if __name__ == "__main__":
    main()
