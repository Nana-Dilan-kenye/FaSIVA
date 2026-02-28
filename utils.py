"""
Utility functions for FaSIVA implementation
"""
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import tarfile
import zipfile
from tqdm import tqdm
import pickle
import json
from datetime import datetime

from config import *

def load_image(image_path):
    """Load image from path"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image

def save_image(image, path):
    """Save image to path"""
    cv2.imwrite(path, image)

def resize_image(image, size=(224, 224)):
    """Resize image to specified size"""
    return cv2.resize(image, size)

def check_resolution(image):
    """Check if image resolution meets threshold"""
    h, w = image.shape[:2]
    return h >= THRESHOLD_RESOLUTION[0] and w >= THRESHOLD_RESOLUTION[1]

def get_resolution(image):
    """Get image resolution"""
    h, w = image.shape[:2]
    return (h, w)

def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors"""
    return np.linalg.norm(vec1 - vec2)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2 + 1e-8)

def normalize_vector(vec):
    """Normalize vector to unit length"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def preprocess_face_image(image, target_size=(224, 224)):
    """Preprocess face image for neural networks"""
    # Resize
    image = cv2.resize(image, target_size)
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image_normalized = image_rgb.astype(np.float32) / 255.0
    
    # Standard normalization for ImageNet models
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_normalized - mean) / std
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(destination)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

def download_lfw_dataset():
    """Use local LFW dataset instead of downloading"""
    global LFW_DATASET_PATH  # <-- must be first
    
    if not os.path.exists(LFW_DATASET_PATH):
        # Check common extracted folder, prioritizing the deepest one
        possible_paths = [
            os.path.join(DATASETS_DIR, "lfw-deepfunneled", "lfw-deepfunneled"), # Most common nested structure
            os.path.join(DATASETS_DIR, "lfw-deepfunneled"),
            os.path.join(DATASETS_DIR, "lfw")
        ]
        found_path = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0: # Check if directory exists and is not empty
                found_path = path
                break # Use the first valid path found (deepest first)
        
        if found_path:
            print(f"LFW dataset found at {found_path}")
            LFW_DATASET_PATH = found_path
            return
        else:
            print(f"LFW dataset not found in {DATASETS_DIR}. Please download and extract it.")
            print("Download URL: http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz")
            print(f"Extract to: {DATASETS_DIR}/lfw-deepfunneled/ (and ensure images are in a nested 'lfw-deepfunneled' folder if applicable).")
            raise FileNotFoundError(
                f"LFW dataset not found. Please download and extract it to {DATASETS_DIR}."
            )
    else:
        print(f"LFW dataset already exists at {LFW_DATASET_PATH}")

def _check_liveness_dataset_structure(dataset_path, dataset_name):
    """Helper to check if liveness dataset has expected subdirectories"""
    if dataset_name == "NUAA":
        real_dir = os.path.join(dataset_path, 'ClientRaw')
        fake_dir = os.path.join(dataset_path, 'ImposterRaw')
        expected_structure_msg = f"Please ensure {dataset_name} is organized with 'ClientRaw' (real) and 'ImposterRaw' (fake) image folders inside {dataset_path}."
    elif dataset_name == "CASIA":
        real_dir = os.path.join(dataset_path, 'Au')
        fake_dir = os.path.join(dataset_path, 'Tp')
        expected_structure_msg = f"Please ensure {dataset_name} is organized with 'Au' (authentic) and 'Tp' (tampered) image folders inside {dataset_path}."
    else: # For Replay-Attack, we expect 'real' and 'fake' directly inside the root after frame extraction
        real_dir = os.path.join(dataset_path, 'real')
        fake_dir = os.path.join(dataset_path, 'fake')
        expected_structure_msg = f"Please ensure {dataset_name} is organized with 'real' and 'fake' image folders directly inside {dataset_path}."
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"Warning: {dataset_name} dataset at {dataset_path} does not have the expected subdirectories.")
        print(expected_structure_msg)
        return False
    print(f"{dataset_name} dataset structure verified at {dataset_path}.")
    return True

def extract_frames_from_video(video_path, output_dir, frame_interval=1):
    """Extracts frames from a video file and saves them as images."""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    frame_count = 0
    saved_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_frame_count} frames from {video_path} to {output_dir}")
    return saved_frame_count


def download_nuaa_dataset():
    """Placeholder for NUAA dataset download and verification"""
    global NUAA_DATASET_PATH
    if not os.path.exists(NUAA_DATASET_PATH):
        print(f"NUAA dataset not found at {NUAA_DATASET_PATH}.")
        print("Please manually download the NUAA dataset.")
        print("Refer to the paper for download sources (e.g., http://parnec.nuaa.edu.cn/xtan/data/NUAAImposterDatabase.zip).")
        print(f"Extract the dataset to: {NUAA_DATASET_PATH}")
        print("Ensure it contains 'ClientRaw' and 'ImposterRaw' subdirectories with images.")
        raise FileNotFoundError(f"NUAA dataset not found. Please download and organize it at {NUAA_DATASET_PATH}.")
    else:
        print(f"NUAA dataset already exists at {NUAA_DATASET_PATH}")
        _check_liveness_dataset_structure(NUAA_DATASET_PATH, "NUAA")

def download_replay_attack_dataset():
    """Prepares Replay-Attack dataset by extracting frames from videos and downloading real samples."""
    global REPLAY_DATASET_PATH
    
    replay_root = REPLAY_DATASET_PATH
    samples_dir = os.path.join(replay_root, 'samples')
    fake_output_dir = os.path.join(replay_root, 'fake')
    real_output_dir = os.path.join(replay_root, 'real')

    if not os.path.exists(replay_root):
        print(f"Replay-Attack dataset not found at {replay_root}.")
        print("Please manually download the Replay-Attack dataset.")
        print("Refer to the paper for download sources (e.g., https://www.idiap.ch/dataset/replayattack).")
        print(f"Extract the dataset to: {replay_root}")
        print(f"Ensure the video files are located in a '{samples_dir}' subdirectory.")
        raise FileNotFoundError(f"Replay-Attack dataset not found. Please download and organize it at {replay_root}.")
    
    print(f"Replay-Attack dataset already exists at {replay_root}")

    # Ensure output directories exist
    os.makedirs(fake_output_dir, exist_ok=True)
    os.makedirs(real_output_dir, exist_ok=True)

    # Extract frames from 'samples' (fake)
    if len(os.listdir(fake_output_dir)) == 0: # Only process if output directory is empty
        print(f"Extracting frames for Replay-Attack fake videos to {fake_output_dir}...")
        video_files = []
        if os.path.exists(samples_dir):
            for root, _, files in os.walk(samples_dir):
                for file in files:
                    if file.lower().endswith(('.mp4', '.mov', '.avi')):
                        video_files.append(os.path.join(root, file))
        
        if not video_files:
            print(f"No video files found in {samples_dir}. Cannot extract fake frames.")
        else:
            for video_file in tqdm(video_files, desc="Extracting Replay-Attack fake frames"):
                extract_frames_from_video(video_file, fake_output_dir, frame_interval=5) # Extract every 5th frame

    # Download and extract frames from provided real video URLs
    real_video_urls = [
        "https://unidata.pro/wp-content/uploads/2024/06/video1.mp4?_=1",
        "https://unidata.pro/wp-content/uploads/2024/06/video2.mp4?_=2"
    ]
    
    if len(os.listdir(real_output_dir)) == 0 and real_video_urls: # Only process if output directory is empty and URLs exist
        print(f"Downloading and extracting frames for Replay-Attack real videos to {real_output_dir}...")
        for i, url in enumerate(tqdm(real_video_urls, desc="Downloading real videos")):
            video_filename = os.path.join(replay_root, f"real_video_{i}.mp4")
            try:
                download_file(url, video_filename)
                extract_frames_from_video(video_filename, real_output_dir, frame_interval=5)
                os.remove(video_filename) # Clean up downloaded video file
            except Exception as e:
                print(f"Error processing real video from {url}: {e}")
    elif not real_video_urls:
        print(f"No real video URLs provided. Please provide URLs to populate {real_output_dir}.")

    _check_liveness_dataset_structure(replay_root, "Replay-Attack")


def download_casia_dataset():
    """Prepares CASIA dataset by extracting frames from videos."""
    global CASIA_DATASET_PATH
    
    casia_root = CASIA_DATASET_PATH
    au_dir = os.path.join(casia_root, 'Au') # Authentic (real) videos/images
    tp_dir = os.path.join(casia_root, 'Tp') # Tampered (fake) videos/images
    
    real_output_dir = os.path.join(casia_root, 'real')
    fake_output_dir = os.path.join(casia_root, 'fake')

    if not os.path.exists(casia_root):
        print(f"CASIA dataset not found at {casia_root}.")
        print("Please manually download the CASIA-FASD dataset.")
        print("Refer to the paper for download sources (e.g., http://www.cbsr.ia.ac.cn/english/CASIA-FASD.asp).")
        print(f"Extract the dataset to: {casia_root}")
        print(f"Ensure the authentic images/videos are in '{au_dir}' and tampered in '{tp_dir}'.")
        raise FileNotFoundError(f"CASIA dataset not found. Please download and organize it at {casia_root}.")
    
    print(f"CASIA dataset already exists at {casia_root}")

    # Ensure output directories exist
    os.makedirs(real_output_dir, exist_ok=True)
    os.makedirs(fake_output_dir, exist_ok=True)

    # Extract frames from 'Au' (real)
    if len(os.listdir(real_output_dir)) == 0: # Only process if output directory is empty
        print(f"Extracting frames for CASIA real data to {real_output_dir}...")
        video_files = []
        image_files = []
        if os.path.exists(au_dir):
            for root, _, files in os.walk(au_dir):
                for file in files:
                    if file.lower().endswith(('.mp4', '.mov', '.avi')):
                        video_files.append(os.path.join(root, file))
                    elif file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                        image_files.append(os.path.join(root, file))
        
        if video_files:
            for video_file in tqdm(video_files, desc="Extracting CASIA real frames"):
                extract_frames_from_video(video_file, real_output_dir, frame_interval=5)
        
        # Copy existing images if any
        if image_files:
            print(f"Copying CASIA real images to {real_output_dir}...")
            for img_file in tqdm(image_files, desc="Copying CASIA real images"):
                # Check if the source file actually exists before linking
                if os.path.exists(img_file):
                    try:
                        os.link(img_file, os.path.join(real_output_dir, os.path.basename(img_file))) # Use hard link to save space
                    except OSError as e:
                        print(f"Warning: Could not hard link {img_file} to {real_output_dir}. Error: {e}. Attempting copy.")
                        import shutil
                        shutil.copy2(img_file, os.path.join(real_output_dir, os.path.basename(img_file)))
                else:
                    print(f"Warning: Source image file not found: {img_file}. Skipping.")

    # Extract frames from 'Tp' (fake)
    if len(os.listdir(fake_output_dir)) == 0: # Only process if output directory is empty
        print(f"Extracting frames for CASIA fake data to {fake_output_dir}...")
        video_files = []
        image_files = []
        if os.path.exists(tp_dir):
            for root, _, files in os.walk(tp_dir):
                for file in files:
                    if file.lower().endswith(('.mp4', '.mov', '.avi')):
                        video_files.append(os.path.join(root, file))
                    elif file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                        image_files.append(os.path.join(root, file))
        
        if video_files:
            for video_file in tqdm(video_files, desc="Extracting CASIA fake frames"):
                extract_frames_from_video(video_file, fake_output_dir, frame_interval=5)
        
        # Copy existing images if any
        if image_files:
            print(f"Copying CASIA fake images to {fake_output_dir}...")
            for img_file in tqdm(image_files, desc="Copying CASIA fake images"):
                # Check if the source file actually exists before linking
                if os.path.exists(img_file):
                    try:
                        os.link(img_file, os.path.join(fake_output_dir, os.path.basename(img_file))) # Use hard link to save space
                    except OSError as e:
                        print(f"Warning: Could not hard link {img_file} to {fake_output_dir}. Error: {e}. Attempting copy.")
                        import shutil
                        shutil.copy2(img_file, os.path.join(fake_output_dir, os.path.basename(img_file)))
                else:
                    print(f"Warning: Source image file not found: {img_file}. Skipping.")

    _check_liveness_dataset_structure(casia_root, "CASIA")


def create_directory_structure():
    """Create necessary directories"""
    dirs = [MODELS_DIR, DATASETS_DIR, 
            os.path.join(DATASETS_DIR, 'train'),
            os.path.join(DATASETS_DIR, 'val'),
            os.path.join(DATASETS_DIR, 'test')]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Directory structure created")

def save_signature(signature, path):
    """Save FaSIVA signature to file"""
    with open(path, 'wb') as f:
        pickle.dump(signature, f)

def load_signature(path):
    """Load FaSIVA signature from file"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def calculate_metrics(tp, fp, tn, fn):
    """Calculate performance metrics (page 11)"""
    metrics = {}
    
    # False Acceptance Rate
    metrics['FAR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # False Rejection Rate
    metrics['FRR'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # True Positive Rate (Sensitivity)
    metrics['TPR'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # True Negative Rate (Specificity)
    metrics['TNR'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Accuracy
    total = tp + fp + tn + fn
    metrics['ACC'] = (tp + tn) / total if total > 0 else 0
    
    return metrics

def get_timestamp():
    """Get current timestamp for logging"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_logging():
    """Setup logging configuration"""
    log_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"fasiva_{get_timestamp()}.log")
    
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
