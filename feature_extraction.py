"""
Feature extraction for FaSIVA signature (page 3)
Extracts F vector (ResNet-50, 2062 dim) and E vector (FaceNet, 512 dim)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
from typing import Tuple

from config import *
from utils import normalize_vector

class FeatureExtractor:
    """Main feature extractor for FaSIVA signature"""
    
    def __init__(self, device=DEVICE):
        self.device = device
        
        # Initialize models
        self.resnet_model = None
        self.facenet_model = None
        
        # Load or create models
        self._load_models()
    
    def _load_models(self):
        """Load or create feature extraction models"""
        print("Initializing feature extractors...")

        # Load ResNet-50 model for F vector (2062 dimensions)
        print("  Loading ResNet-50 for identification...")
        
        if os.path.exists(RESNET_MODEL_PATH):
            print(f"    Loading pre-trained ResNet-50 from {RESNET_MODEL_PATH}")
            resnet_base = models.resnet50(pretrained=False) # Initialize without pretrained weights
            self.resnet_model = nn.Sequential(*list(resnet_base.children())[:-1])
            self.resnet_model = nn.Sequential(
                self.resnet_model,
                nn.Flatten(),
                nn.Linear(2048, RESNET_FEATURES_DIM),
                nn.BatchNorm1d(RESNET_FEATURES_DIM),
                nn.ReLU()
            )
            state_dict = torch.load(RESNET_MODEL_PATH, map_location=self.device)
            # Handle potential nested state_dict
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.resnet_model.load_state_dict(state_dict)
        else:
            print("    No pre-trained ResNet-50 found. Initializing with ImageNet weights and adding custom head.")
            # Initialize ResNet-50 with ImageNet pretrained weights
            base_resnet_pretrained = models.resnet50(pretrained=True)
            
            # Create the feature extractor model structure
            self.resnet_model = nn.Sequential(*list(base_resnet_pretrained.children())[:-1]) # Base ResNet
            self.resnet_model = nn.Sequential(
                self.resnet_model,
                nn.Flatten(),
                nn.Linear(2048, RESNET_FEATURES_DIM),
                nn.BatchNorm1d(RESNET_FEATURES_DIM),
                nn.ReLU()
            )
            # The base part already has pretrained weights. The new Linear and BatchNorm layers will be randomly initialized.
        
        self.resnet_model.to(self.device)
        self.resnet_model.eval()
        print(f"    ✓ ResNet-50 ready (output: {RESNET_FEATURES_DIM}D)")
        
        # Load FaceNet model for E vector (use standard 512 dimensions)
        print("  Loading FaceNet for verification...")
        if os.path.exists(FACENET_MODEL_PATH):
            print(f"    Loading pre-trained FaceNet from {FACENET_MODEL_PATH}")
            self.facenet_model = InceptionResnetV1(
                pretrained=None, # Initialize without pretrained weights
                classify=False
            )
            state_dict = torch.load(FACENET_MODEL_PATH, map_location=self.device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # Filter out unexpected keys (e.g., 'logits.weight', 'logits.bias')
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('logits.')}
            self.facenet_model.load_state_dict(filtered_state_dict)
        else:
            print("    No pre-trained FaceNet found. Initializing with VGGFace2 weights.")
            self.facenet_model = InceptionResnetV1(
                pretrained='vggface2',
                classify=False  # Get embeddings, not classifications
            )
        
        self.facenet_model.to(self.device)
        self.facenet_model.eval()
        
        # Add a linear layer to reduce FaceNet output to 128 dimensions as per paper
        self.facenet_output_layer = nn.Linear(512, FACENET_FEATURES_DIM).to(self.device)
        
        print(f"    ✓ FaceNet ready (output: {FACENET_FEATURES_DIM}D)")
        
        print("✓ Feature extractors initialized successfully")

    def save_models(self):
        """Save model weights for ResNet-50 and FaceNet"""
        torch.save(self.resnet_model.state_dict(), RESNET_MODEL_PATH)
        torch.save(self.facenet_model.state_dict(), FACENET_MODEL_PATH)
        print(f"Feature extractor models saved to {MODELS_DIR}")
    
    def _preprocess_for_resnet(self, face_image: np.ndarray) -> torch.Tensor:
        """Preprocess image for ResNet-50"""
        # Resize to 224x224
        face_resized = cv2.resize(face_image, (224, 224))
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        face_normalized = (face_normalized - mean) / std
        
        # Convert to tensor
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1)
        face_tensor = face_tensor.unsqueeze(0).to(self.device,dtype=torch.float)
        
        return face_tensor
    
    def _preprocess_for_facenet(self, face_image: np.ndarray) -> torch.Tensor:
        """Preprocess image for FaceNet"""
        # Resize to 160x160
        face_resized = cv2.resize(face_image, (160, 160))
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1] (FaceNet expects this)
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
        
        # Convert to tensor
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1)
        face_tensor = face_tensor.unsqueeze(0).to(self.device,dtype=torch.float)
        
        return face_tensor
    
    def extract_f_vector(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract identification vector F (2062 dimensions)
        Using ResNet-50 as per paper (page 4)
        """
        # Preprocess for ResNet
        image_tensor = self._preprocess_for_resnet(face_image)
        
        # Extract features
        with torch.no_grad():
            features = self.resnet_model(image_tensor)
        
        # Convert to numpy and flatten
        f_vector = features.cpu().numpy().flatten()
        
        # Ensure correct dimension
        if len(f_vector) != RESNET_FEATURES_DIM:
            print(f"Warning: F vector has {len(f_vector)} dimensions, expected {RESNET_FEATURES_DIM}")
            # Pad or truncate if necessary
            if len(f_vector) > RESNET_FEATURES_DIM:
                f_vector = f_vector[:RESNET_FEATURES_DIM]
            else:
                padding = np.zeros(RESNET_FEATURES_DIM - len(f_vector))
                f_vector = np.concatenate([f_vector, padding])
        
        return normalize_vector(f_vector)
    
    def extract_e_vector(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract verification vector E (512 dimensions from FaceNet)
        Updated to use standard FaceNet output
        """
        # Preprocess for FaceNet
        image_tensor = self._preprocess_for_facenet(face_image)
        
        # Extract features
        with torch.no_grad():
            features_512d = self.facenet_model(image_tensor)
            # Apply the new linear layer to get 128 dimensions
            features_128d = self.facenet_output_layer(features_512d)
        
        # Convert to numpy and flatten
        e_vector = features_128d.cpu().numpy().flatten()
        
        # FaceNet outputs 128 dimensions now
        expected_dim = FACENET_FEATURES_DIM
        if len(e_vector) != expected_dim:
            print(f"Warning: E vector has {len(e_vector)} dimensions, expected {expected_dim}")
            # Pad or truncate if necessary
            if len(e_vector) > expected_dim:
                e_vector = e_vector[:expected_dim]
            else:
                padding = np.zeros(expected_dim - len(e_vector))
                e_vector = np.concatenate([e_vector, padding])
        
        return normalize_vector(e_vector)
    
    def extract_signature_features(self, face_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract both F and E vectors for FaSIVA signature
        Returns: (F_vector, E_vector)
        """
        f_vector = self.extract_f_vector(face_image)
        e_vector = self.extract_e_vector(face_image)
        
        return f_vector, e_vector
    
    def batch_extract(self, face_images: list) -> Tuple[list, list]:
        """
        Extract features for multiple faces
        Returns: (F_vectors, E_vectors)
        """
        f_vectors = []
        e_vectors = []
        
        for face_image in face_images:
            try:
                f_vec, e_vec = self.extract_signature_features(face_image)
                f_vectors.append(f_vec)
                e_vectors.append(e_vec)
            except Exception as e:
                print(f"Error extracting features: {e}")
                # Append zero vectors as fallback
                f_vectors.append(np.zeros(RESNET_FEATURES_DIM))
                e_vectors.append(np.zeros(FACENET_FEATURES_DIM))  # FaceNet dimension
        
        return f_vectors, e_vectors


# Global feature extractor instance
feature_extractor = FeatureExtractor()
