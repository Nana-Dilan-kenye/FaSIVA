"""
Configuration file for FaSIVA implementation
All parameters from the paper are defined here
"""
import os
import torch

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATASETS_DIR = os.path.join(BASE_DIR, 'dataset')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

# Parameters from paper
THRESHOLD_RESOLUTION = (35, 35)  # Minimum acceptable resolution (page 4)
# THRESHOLD_RESOLUTION = (90,90)  
SUPER_RES_FACTOR = 4             # k=4 from paper (page 4)
FACE_DETECTION_CONFIDENCE = 0.99 # MTCNN confidence threshold

# Feature extraction dimensions (page 3)
RESNET_FEATURES_DIM = 2062       # Paper mentions 2062 dimensions for F vector
FACENET_FEATURES_DIM = 128       # 128 dimensions for E vector from paper

# Liveness detection thresholds
EYE_BLINK_THRESHOLD = 0.3        # EAR threshold (page 3, equation 8)
LIVENESS_REFLECTION_THRESHOLD = 0.5  # Reflection coefficient threshold

# Verification thresholds
VERIFICATION_THRESHOLD = 0.5     # Euclidean distance threshold for verification
IDENTIFICATION_THRESHOLD = 0.7   # Threshold for person identification

# Model paths
RESNET_MODEL_PATH = os.path.join(MODELS_DIR, 'resnet50_fasiva.pth')
FACENET_MODEL_PATH = os.path.join(MODELS_DIR, 'facenet_fasiva.pth')
FSRCNN_MODEL_PATH = os.path.join(MODELS_DIR, 'fsrcnn_x4.pth')
LIVENESS_MODEL_PATH = os.path.join(MODELS_DIR, 'liveness_alex.pth')

# Dataset paths
LFW_DATASET_PATH = os.path.join(DATASETS_DIR, "lfw-deepfunneled", "lfw-deepfunneled")
NUAA_DATASET_PATH = os.path.join(DATASETS_DIR, 'nuaa')
REPLAY_DATASET_PATH = os.path.join(DATASETS_DIR, 'replay_attack_dataset') # Updated path
CASIA_DATASET_PATH = os.path.join(DATASETS_DIR, 'CASIA2') # Updated path

# Database
DATABASE_PATH = os.path.join(BASE_DIR, 'faces_database.db')

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 2 # Changed from 20 to 2 as per user request
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Blink detection (page 11)
BLINK_TIME_WINDOW = 3.0  # Seconds to wait for a blink
MIN_EYE_CLOSE_RATIO = 0.2  # Minimum EAR for eye closure detection
