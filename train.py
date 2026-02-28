"""
Training script for FaSIVA components
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.models as models # Added import for models

from config import *
from super_resolution import FSRCNN, train_fsrcnn, FaceSRDataset
from feature_extraction import FeatureExtractor
from liveness_detection import CNNLivenessDetector
from utils import download_lfw_dataset, download_nuaa_dataset, download_replay_attack_dataset, download_casia_dataset


def train_super_resolution():
    """Train FSRCNN model for face super-resolution"""
    print("Training Super-Resolution Model...")
    print("=" * 50)
    
    from super_resolution import prepare_sr_training
    
    # This function already handles:
    # 1. Downloading LFW
    # 2. Creating dataset
    # 3. Splitting train/val
    # 4. Creating dataloaders
    # 5. Creating model and training it
    # 6. Saving model
    model, train_losses, val_losses = prepare_sr_training()
    
    print(f"\nTraining completed!")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"Model saved to {FSRCNN_MODEL_PATH}")


class FaceIdentityDataset(Dataset):
    """Dataset for training feature extractors on face identities (e.g., LFW)"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = []

        # Collect images and assign labels
        label_idx = 0
        for person_name in os.listdir(root_dir):
            person_dir = os.path.join(root_dir, person_name)
            if os.path.isdir(person_dir):
                if person_name not in self.class_to_idx:
                    self.class_to_idx[person_name] = label_idx
                    self.idx_to_class.append(person_name)
                    label_idx += 1
                
                current_label = self.class_to_idx[person_name]
                for img_name in os.listdir(person_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(person_dir, img_name))
                        self.labels.append(current_label)
        
        print(f"Found {len(self.image_paths)} images for {len(self.class_to_idx)} identities in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for corrupted images
            print(f"Warning: Could not load image {img_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self)) # Load next image
        
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label


class FilteredFaceIdentityDataset(Dataset):
    """Dataset for training/evaluation feature extractors on face identities (e.g., LFW), with filtering."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {img_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def train_feature_extractors():
    """Train feature extraction models on face dataset"""
    print("Training Feature Extraction Models...")
    print("=" * 50)
    
    # Initialize FeatureExtractor (loads pre-trained ResNet-50 and FaceNet)
    feature_extractor_instance = FeatureExtractor()
    
    # For ResNet-50 (F vector) training:
    # We need to train the custom classification head on top of ResNet-50
    # The paper mentions SoftMax regression layer for identification.
    
    print("Training ResNet-50 for identification (F vector)...")
    
    # Prepare dataset (using LFW for demonstration as it has identities)
    # Note: LFW is primarily for verification, but can be adapted for identification
    # by treating each person as a class.
    
    # Download LFW if not already done
    download_lfw_dataset()
    
    # Define a simple transform for ResNet-50 input
    def resnet_transform(image):
        face_resized = cv2.resize(image, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        face_normalized = (face_normalized - mean) / std
        return torch.from_numpy(face_normalized).permute(2, 0, 1)

    full_dataset = FaceIdentityDataset(LFW_DATASET_PATH, transform=resnet_transform)
    
    # Filter out classes with too few samples for training
    class_counts = defaultdict(int)
    for label in full_dataset.labels:
        class_counts[label] += 1
    
    # Only include classes with more than 1 sample for training
    # This is a heuristic to ensure meaningful training
    valid_indices = [i for i, label in enumerate(full_dataset.labels) if class_counts[label] > 1]
    
    filtered_image_paths = [full_dataset.image_paths[i] for i in valid_indices]
    filtered_labels = [full_dataset.labels[i] for i in valid_indices]
    
    # Re-map labels to be contiguous after filtering
    unique_labels = sorted(list(set(filtered_labels)))
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    remapped_labels = [label_mapping[old_label] for old_label in filtered_labels]

    # Create a new dataset with filtered and remapped data
    train_val_dataset = FilteredFaceIdentityDataset(filtered_image_paths, remapped_labels, transform=resnet_transform)

    if len(train_val_dataset) == 0:
        print("No sufficient data for ResNet-50 training after filtering. Skipping.")
        feature_extractor_instance.save_models()
        print(f"Models saved to {MODELS_DIR}")
        return

    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Modify the ResNet-50 model to have a classification head for training
    # The original ResNet-50 in FeatureExtractor has a custom linear layer for 2062 features.
    # For training, we need a classification head with `num_classes`.
    
    num_classes = len(unique_labels)
    
    # Temporarily replace the last layers of ResNet-50 for classification training
    # We'll save the feature extraction part later
    
    # Get the base ResNet-50 without the custom linear layer
    base_resnet = models.resnet50(pretrained=True)
    base_resnet = nn.Sequential(*list(base_resnet.children())[:-1]) # Remove original FC layer
    
    # Add a new classification head
    resnet_classifier = nn.Sequential(
        base_resnet,
        nn.Flatten(),
        nn.Linear(2048, num_classes) # Output to number of identities
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_classifier.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print(f"Training ResNet-50 classifier for {num_classes} identities...")
    for epoch in range(NUM_EPOCHS):
        resnet_classifier.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in tqdm(train_loader, desc=f"ResNet-50 Epoch {epoch+1}/{NUM_EPOCHS} (Train)"):
            inputs, labels = inputs.to(DEVICE, dtype=torch.float32), labels.to(DEVICE) # Cast inputs to float32
            optimizer.zero_grad()
            outputs = resnet_classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = correct_predictions / total_samples
        
        # Validation
        resnet_classifier.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"ResNet-50 Epoch {epoch+1}/{NUM_EPOCHS} (Val)"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = resnet_classifier(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()
        
        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_accuracy = val_correct_predictions / val_total_samples
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.4f}")
        
        scheduler.step()

    print("ResNet-50 identification training complete.")
    
    # After training the classifier, we need to save the feature extraction part
    # which is the base_resnet + the custom linear layer for RESNET_FEATURES_DIM
    
    # Reconstruct the feature extractor part of ResNet-50
    feature_extractor_instance.resnet_model = nn.Sequential(
        base_resnet,
        nn.Flatten(),
        nn.Linear(2048, RESNET_FEATURES_DIM),
        nn.BatchNorm1d(RESNET_FEATURES_DIM),
        nn.ReLU()
    ).to(DEVICE)
    
    # Copy weights from the trained base_resnet to the feature extractor
    feature_extractor_instance.resnet_model[0].load_state_dict(base_resnet.state_dict())
    
    # Save the updated feature extractor models
    feature_extractor_instance.save_models()
    
    print(f"Models saved to {MODELS_DIR}")


def evaluate_models():
    """Evaluate the trained models and print their accuracies."""
    print("Evaluating Models...")
    print("=" * 50)

    # 1. Evaluate ResNet-50 for identification
    print("Evaluating ResNet-50 for identification...")
    feature_extractor_instance = FeatureExtractor() # Loads trained models
    resnet_model = feature_extractor_instance.resnet_model
    
    # Prepare LFW dataset for evaluation
    def resnet_transform(image):
        face_resized = cv2.resize(image, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        face_normalized = (face_normalized - mean) / std
        return torch.from_numpy(face_normalized).permute(2, 0, 1)

    full_dataset = FaceIdentityDataset(LFW_DATASET_PATH, transform=resnet_transform)
    
    # Filter out classes with too few samples for evaluation (same as training)
    class_counts = defaultdict(int)
    for label in full_dataset.labels:
        class_counts[label] += 1
    
    valid_indices = [i for i, label in enumerate(full_dataset.labels) if class_counts[label] > 1]
    filtered_image_paths = [full_dataset.image_paths[i] for i in valid_indices]
    filtered_labels = [full_dataset.labels[i] for i in valid_indices]
    
    unique_labels = sorted(list(set(filtered_labels)))
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    remapped_labels = [label_mapping[old_label] for old_label in filtered_labels]

    eval_dataset = FilteredFaceIdentityDataset(filtered_image_paths, remapped_labels, transform=resnet_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if len(eval_dataset) == 0:
        print("No sufficient data for ResNet-50 evaluation after filtering. Skipping.")
    else:
        resnet_model.eval()
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in tqdm(eval_loader, desc="ResNet-50 Evaluation"):
                inputs, labels = inputs.to(DEVICE, dtype=torch.float32), labels.to(DEVICE)
                outputs = resnet_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        accuracy = correct_predictions / total_samples
        print(f"ResNet-50 Evaluation Accuracy: {accuracy:.4f}")

    # 2. Evaluate CNNLivenessDetector
    print("\nEvaluating CNNLivenessDetector...")
    detector = CNNLivenessDetector() # Loads trained model
    
    # Prepare liveness dataset for evaluation
    def alexnet_transform(image):
        image_resized = cv2.resize(image, (64, 64))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_norm = (image_norm - mean) / std
        return torch.from_numpy(image_norm).permute(2, 0, 1)

    liveness_root_dirs = [NUAA_DATASET_PATH, CASIA_DATASET_PATH, REPLAY_DATASET_PATH] # Include Replay-Attack for evaluation
    full_liveness_dataset = LivenessSpoofDataset(liveness_root_dirs, transform=alexnet_transform)
    eval_loader = DataLoader(full_liveness_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if len(full_liveness_dataset) == 0:
        print("No liveness detection data found for evaluation. Skipping.")
    else:
        detector.model.eval()
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in tqdm(eval_loader, desc="Liveness Detector Evaluation"):
                inputs, labels = inputs.to(DEVICE, dtype=torch.float32), labels.to(DEVICE)
                outputs = detector.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        accuracy = correct_predictions / total_samples
        print(f"CNNLivenessDetector Evaluation Accuracy: {accuracy:.4f}")

    print("\nEvaluation complete!")


class LivenessSpoofDataset(Dataset):
    """
    Dataset for liveness detection training using NUAA, Replay-Attack, CASIA datasets.
    Combines images from 'real' and 'fake' subdirectories.
    """
    def __init__(self, root_dirs, transform=None):
        self.root_dirs = root_dirs if isinstance(root_dirs, list) else [root_dirs]
        self.transform = transform
        self.image_paths = []
        self.labels = [] # 0 for real, 1 for fake

        for root_dir in self.root_dirs:
            # Handle specific dataset structures
            if os.path.basename(root_dir) == 'nuaa':
                real_sub_dir = 'ClientRaw'
                fake_sub_dir = 'ImposterRaw'
            elif os.path.basename(root_dir) == 'CASIA2':
                real_sub_dir = 'Au' # Authentic images
                fake_sub_dir = 'Tp' # Tampered images
            elif os.path.basename(root_dir) == 'replay_attack_dataset':
                # Replay-Attack dataset typically has 'attack' and 'live' folders, or similar.
                # Based on common structures, 'live' is real, 'attack' is fake.
                # If the user has extracted frames into 'real' and 'fake' directly, use those.
                # Otherwise, we need to check the actual structure.
                # For now, assuming user will create 'real' and 'fake' inside replay_attack_dataset
                real_sub_dir = 'real'
                fake_sub_dir = 'fake'
            else:
                real_sub_dir = 'real'
                fake_sub_dir = 'fake'

            real_dir = os.path.join(root_dir, real_sub_dir)
            fake_dir = os.path.join(root_dir, fake_sub_dir)

            if os.path.exists(real_dir):
                for img_name in os.listdir(real_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(real_dir, img_name))
                        self.labels.append(0) # Real
            
            if os.path.exists(fake_dir):
                for img_name in os.listdir(fake_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(fake_dir, img_name))
                        self.labels.append(1) # Fake
        
        print(f"Found {len(self.image_paths)} images for liveness detection across {len(self.root_dirs)} datasets.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {img_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))
        
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_liveness_detector():
    """Train CNN-based liveness detector"""
    print("Training Liveness Detection Model...")
    print("=" * 50)
    
    detector = CNNLivenessDetector()
    
    # Prepare dataset (NUAA, Replay-Attack, CASIA)
    print("Preparing Liveness Detection Datasets...")
    download_nuaa_dataset()
    download_replay_attack_dataset()
    download_casia_dataset()

    # Define a simple transform for AlexNet input
    def alexnet_transform(image):
        # Preprocessing as in CNNLivenessDetector.preprocess
        image_resized = cv2.resize(image, (64, 64))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_norm = (image_norm - mean) / std
        return torch.from_numpy(image_norm).permute(2, 0, 1)

    # Combine all datasets for liveness detection training
    liveness_root_dirs = [NUAA_DATASET_PATH, REPLAY_DATASET_PATH, CASIA_DATASET_PATH]
    full_liveness_dataset = LivenessSpoofDataset(liveness_root_dirs, transform=alexnet_transform)

    if len(full_liveness_dataset) == 0:
        print("No liveness detection data found. Skipping training.")
        detector.save_model(LIVENESS_MODEL_PATH)
        print(f"Model structure saved to {LIVENESS_MODEL_PATH}")
        return

    train_size = int(0.8 * len(full_liveness_dataset))
    val_size = len(full_liveness_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_liveness_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(detector.model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("Training CNNLivenessDetector...")
    for epoch in range(NUM_EPOCHS):
        detector.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in tqdm(train_loader, desc=f"Liveness Epoch {epoch+1}/{NUM_EPOCHS} (Train)"):
            inputs, labels = inputs.to(DEVICE, dtype=torch.float32), labels.to(DEVICE) # Cast inputs to float32
            optimizer.zero_grad()
            outputs = detector.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = correct_predictions / total_samples
        
        # Validation
        detector.model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Liveness Epoch {epoch+1}/{NUM_EPOCHS} (Val)"):
                inputs, labels = inputs.to(DEVICE, dtype=torch.float32), labels.to(DEVICE) # Cast inputs to float32
                outputs = detector.model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()
        
        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_accuracy = val_correct_predictions / val_total_samples
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.4f}")
        
        scheduler.step()

    print("Liveness detection training complete.")
    detector.save_model(LIVENESS_MODEL_PATH)
    
    print(f"Model saved to {MODELS_DIR}")


def prepare_training_data():
    """Prepare all training data"""
    print("Preparing Training Data...")
    print("=" * 50)
    
    # Create directories
    os.makedirs(os.path.join(DATASETS_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATASETS_DIR, 'val'), exist_ok=True)
    os.makedirs(os.path.join(DATASETS_DIR, 'test'), exist_ok=True)
    
    # Download LFW dataset for super-resolution and feature extractor training
    download_lfw_dataset()
    
    # Download spoofing datasets for liveness detection
    download_nuaa_dataset()
    download_replay_attack_dataset()
    download_casia_dataset()
    
    print("Training data preparation complete.")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train FaSIVA components')
    parser.add_argument('--component', type=str, default='all',
                       choices=['all', 'super_res', 'features', 'liveness', 'prepare', 'evaluate'],
                       help='Component to train or evaluate')
    
    args = parser.parse_args()
    
    print("FaSIVA Training Script")
    print("=" * 60)
    
    if args.component in ['all', 'prepare']:
        prepare_training_data()
    
    if args.component in ['all', 'super_res']:
        if os.path.exists(FSRCNN_MODEL_PATH):
            print(f"FSRCNN model already exists at {FSRCNN_MODEL_PATH}. Skipping training.")
        else:
            train_super_resolution()
    
    if args.component in ['all', 'features']:
        train_feature_extractors()
    
    if args.component in ['all', 'liveness']:
        train_liveness_detector()
    
    if args.component == 'evaluate':
        evaluate_models()

    print("\nOperation complete!")
    print("All models saved to:", MODELS_DIR)


if __name__ == "__main__":
    main()
