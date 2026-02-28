"""
Super-resolution module using FSRCNN (page 4)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from PIL import Image
import glob
from tqdm import tqdm

from config import *
from utils import download_lfw_dataset

class FSRCNN(nn.Module):
    """
    Fast Super-Resolution Convolutional Neural Network
    Adapted for face images as per paper (page 4)
    """
    
    def __init__(self, scale_factor=4, num_channels=1, d=56, s=12, m=4):
        """
        Args:
            scale_factor: Super-resolution factor (k=4 in paper)
            num_channels: Input channels (1 for grayscale)
            d: Number of feature dimension
            s: Number of shrinking filters
            m: Number of mapping layers
        """
        super(FSRCNN, self).__init__()
        self.scale_factor = scale_factor
        
        # Feature extraction (page 4, equation 5)
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2),
            nn.PReLU()
        )
        
        # Shrinking (page 4)
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU()
        )
        
        # Mapping (m layers) (page 4, equation 4)
        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mapping_layers.append(nn.PReLU())
        self.mapping = nn.Sequential(*mapping_layers)
        
        # Expanding (page 4)
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU()
        )
        
        # Deconvolution (page 4, equation 3)
        self.deconv = nn.ConvTranspose2d(
            d, num_channels, kernel_size=9,
            stride=scale_factor, padding=4,
            output_padding=scale_factor-1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight.data, mean=0.0, std=0.001)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Feature extraction: F1(Y) = ω1 × Y + B1
        x = self.feature_extraction(x)
        
        # Shrinking
        x = self.shrinking(x)
        
        # Mapping: F2(Y) = f(ω2 × F1(Y) + B2)
        x = self.mapping(x)
        
        # Expanding
        x = self.expanding(x)
        
        # Deconvolution: F(Y) = ω3 × F2(Y) + B3
        x = self.deconv(x)
        
        return x
    
    def save_model(self, path):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'scale_factor': self.scale_factor,
            'config': {
                'd': self.feature_extraction[0].out_channels,
                's': self.shrinking[0].out_channels,
                'm': len(self.mapping) // 2
            }
        }, path)
    
    @classmethod
    def load_model(cls, path, device=DEVICE):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            scale_factor=checkpoint['scale_factor'],
            d=checkpoint['config']['d'],
            s=checkpoint['config']['s'],
            m=checkpoint['config']['m']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model


class FaceSRDataset(Dataset):
    """Dataset for super-resolution training on face images"""
    
    def __init__(self, lfw_path, scale_factor=4, patch_size=35):
        """
        Args:
            lfw_path: Path to LFW dataset
            scale_factor: Super-resolution factor
            patch_size: Size of low-resolution patches
        """
        self.lfw_path = lfw_path
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.hr_size = patch_size * scale_factor
        
        # Collect all face images
        self.image_paths = []
        for root, dirs, files in os.walk(lfw_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(self.image_paths)} face images for SR training")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            # Return a random image if loading fails
            return self.__getitem__((idx + 1) % len(self))
        
        # Random crop to HR size
        h, w = image.shape
        if h < self.hr_size or w < self.hr_size:
            # Resize if too small
            image = cv2.resize(image, (self.hr_size, self.hr_size))
            h, w = self.hr_size, self.hr_size
        
        # Random crop
        top = np.random.randint(0, h - self.hr_size)
        left = np.random.randint(0, w - self.hr_size)
        hr_patch = image[top:top+self.hr_size, left:left+self.hr_size]
        
        # Create LR patch by downscaling
        lr_patch = cv2.resize(hr_patch, (self.patch_size, self.patch_size),
                             interpolation=cv2.INTER_CUBIC)
        
        # Normalize to [0, 1]
        hr_patch = hr_patch.astype(np.float32) / 255.0
        lr_patch = lr_patch.astype(np.float32) / 255.0
        
        # Add channel dimension
        hr_patch = np.expand_dims(hr_patch, axis=0)
        lr_patch = np.expand_dims(lr_patch, axis=0)
        
        return torch.FloatTensor(lr_patch), torch.FloatTensor(hr_patch)


def train_fsrcnn(model, train_loader, val_loader, num_epochs=20):
    """Train FSRCNN model"""
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for lr_imgs, hr_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                outputs = model(lr_imgs)
                loss = criterion(outputs, hr_imgs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    return train_losses, val_losses


def apply_super_resolution(image, model=None):
    """
    Apply super-resolution to an image
    Args:
        image: Input BGR image
        model: Pre-trained FSRCNN model (will load if None)
    Returns:
        Super-resolved image
    """
    if model is None:
        if not os.path.exists(FSRCNN_MODEL_PATH):
            raise FileNotFoundError(f"FSRCNN model not found at {FSRCNN_MODEL_PATH}")
        model = FSRCNN.load_model(FSRCNN_MODEL_PATH, DEVICE)
    
    # Convert to YCrCb (FSRCNN works on Y channel)
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0].astype(np.float32) / 255.0
        cr_channel = ycrcb[:, :, 1]
        cb_channel = ycrcb[:, :, 2]
    else:
        y_channel = image.astype(np.float32) / 255.0
        cr_channel = cb_channel = None
    
    # Prepare input tensor
    y_tensor = torch.FloatTensor(y_channel).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    # Apply super-resolution
    with torch.no_grad():
        sr_y_tensor = model(y_tensor)
    
    # Convert back to numpy
    sr_y = sr_y_tensor.squeeze().cpu().numpy() * 255.0
    sr_y = np.clip(sr_y, 0, 255).astype(np.uint8)
    
    # Upscale chroma channels using bicubic interpolation
    if cr_channel is not None and cb_channel is not None:
        h, w = sr_y.shape
        sr_cr = cv2.resize(cr_channel, (w, h), interpolation=cv2.INTER_CUBIC)
        sr_cb = cv2.resize(cb_channel, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Merge channels and convert back to BGR
        sr_ycrcb = cv2.merge([sr_y, sr_cr, sr_cb])
        sr_image = cv2.cvtColor(sr_ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        sr_image = sr_y
    
    return sr_image


def prepare_sr_training():
    """Prepare and train FSRCNN model on face images"""
    # Download dataset if not exists
    download_lfw_dataset()
    
    # Create dataset
    dataset = FaceSRDataset(LFW_DATASET_PATH, scale_factor=SUPER_RES_FACTOR)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    model = FSRCNN(scale_factor=SUPER_RES_FACTOR).to(DEVICE)
    
    # Train model
    print("Training FSRCNN on face images...")
    train_losses, val_losses = train_fsrcnn(model, train_loader, val_loader, NUM_EPOCHS)
    
    # Save model
    model.save_model(FSRCNN_MODEL_PATH)
    print(f"Model saved to {FSRCNN_MODEL_PATH}")
    
    return model, train_losses, val_losses


# Global FSRCNN model instance
fsrcnn_model = None

def get_fsrcnn_model():
    """Get or load FSRCNN model"""
    global fsrcnn_model
    if fsrcnn_model is None:
        if os.path.exists(FSRCNN_MODEL_PATH):
            fsrcnn_model = FSRCNN.load_model(FSRCNN_MODEL_PATH, DEVICE)
        else:
            raise FileNotFoundError(f"FSRCNN model not found. Train it first.")
    return fsrcnn_model