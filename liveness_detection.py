"""
Liveness detection module for FaSIVA (page 8-11)
Includes reflection detection (eq 7) and eye blink detection (eq 8, 15)
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import dlib
from scipy.spatial import distance as dist
from typing import Tuple, List, Optional

from config import *
from utils import get_timestamp

class ReflectionDetector:
    """
    Reflection-based liveness detection (page 3, equation 7)
    I(x,y) = f_c(x,y) × ρ(x,y) × A_light × cosθ
    """
    
    def __init__(self, threshold=LIVENESS_REFLECTION_THRESHOLD):
        self.threshold = threshold
    
    def calculate_reflection_coefficient(self, image: np.ndarray) -> float:
        """
        Calculate reflection coefficient ρ(x,y)
        Simplified implementation based on paper
        """
        if len(image.shape) != 3:
            # If not a color image, convert to BGR for consistency
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Convert to YCrCb color space for luminance and chrominance analysis
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)

        # 1. Luminance (Y channel) variation
        # Real faces tend to have more natural luminance variations
        y_hist = cv2.calcHist([Y], [0], None, [256], [0, 256])
        y_hist = y_hist.flatten() / y_hist.sum()
        y_entropy = -np.sum(y_hist * np.log(y_hist + 1e-8))
        y_variance = np.var(Y) / 255.0 # Normalize variance

        # 2. Chrominance (Cr, Cb channels) variation
        # Spoofs might have less natural color distribution
        cr_hist = cv2.calcHist([Cr], [0], None, [256], [0, 256])
        cr_hist = cr_hist.flatten() / cr_hist.sum()
        cr_entropy = -np.sum(cr_hist * np.log(cr_hist + 1e-8))

        cb_hist = cv2.calcHist([Cb], [0], None, [256], [0, 256])
        cb_hist = cb_hist.flatten() / cb_hist.sum()
        cb_entropy = -np.sum(cb_hist * np.log(cb_hist + 1e-8))

        # 3. Gradient magnitude (edge information)
        # Real faces have richer texture and edge details
        sobelx = cv2.Sobel(Y, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(Y, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_mean = np.mean(gradient_magnitude) / 255.0 # Normalize

        # 4. Image sharpness (using Laplacian variance)
        # Spoofs (especially printed) might lack sharpness or have artificial sharpness
        laplacian_var = cv2.Laplacian(Y, cv2.CV_64F).var()
        # Normalize laplacian_var (empirical scaling, adjust as needed)
        laplacian_norm = min(1.0, laplacian_var / 1000.0) 

        # Combine features into a reflection coefficient
        # Weights are empirical and can be fine-tuned
        reflection_coeff = (
            0.2 * (y_entropy / np.log(256)) +  # Normalized Y entropy
            0.1 * y_variance +                 # Normalized Y variance
            0.15 * (cr_entropy / np.log(256)) + # Normalized Cr entropy
            0.15 * (cb_entropy / np.log(256)) + # Normalized Cb entropy
            0.2 * gradient_mean +              # Normalized gradient mean
            0.2 * laplacian_norm               # Normalized sharpness
        )
        
        return reflection_coeff
    
    def detect(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if image shows a live face based on reflection
        Returns: (is_live, confidence)
        """
        reflection_coeff = self.calculate_reflection_coefficient(image)
        is_live = reflection_coeff > self.threshold
        confidence = abs(reflection_coeff - self.threshold)
        
        return is_live, reflection_coeff


class EyeBlinkDetector:
    """
    Eye blink detection using facial landmarks (page 3, equation 8)
    EAR = (||p2 - p6|| - ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    
    def __init__(self, ear_threshold=EYE_BLINK_THRESHOLD):
        self.ear_threshold = ear_threshold
        
        # Initialize dlib face detector and shape predictor
        try:
            self.detector = dlib.get_frontal_face_detector()
            predictor_path = os.path.join(MODELS_DIR, 'shape_predictor_68_face_landmarks.dat')
            
            if not os.path.exists(predictor_path):
                # Try to download if not exists
                self._download_shape_predictor(predictor_path)
            
            self.predictor = dlib.shape_predictor(predictor_path)
        except Exception as e:
            print(f"Warning: Could not initialize dlib: {e}")
            self.detector = None
            self.predictor = None
    
    def _download_shape_predictor(self, path):
        """Download dlib shape predictor if not exists"""
        print("Downloading facial landmark predictor...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        
        import requests, bz2
        response = requests.get(url, stream=True)
        
        with open(path + '.bz2', 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        
        # Extract
        with bz2.open(path + '.bz2', 'rb') as f:
            data = f.read()
        
        with open(path, 'wb') as f:
            f.write(data)
        
        os.remove(path + '.bz2')
        print(f"Shape predictor downloaded to {path}")
    
    def eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR)
        eye_points: Array of 6 (x,y) coordinates for one eye
        """
        # Vertical distances
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        
        # Horizontal distance
        C = dist.euclidean(eye_points[0], eye_points[3])
        
        # Calculate EAR (equation 8)
        ear = (A + B) / (2.0 * C + 1e-8)
        
        return ear
    
    def get_eye_landmarks(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get eye landmarks from face image
        Returns: (left_eye_points, right_eye_points) or None if no face detected
        """
        if self.detector is None or self.predictor is None:
            return None
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return None
        
        # Use the first face
        face = faces[0]
        
        # Get facial landmarks
        landmarks = self.predictor(gray, face)
        landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                             for i in range(68)])
        
        # Extract eye regions (indices from dlib 68-point model)
        left_eye_indices = list(range(36, 42))  # 36-41
        right_eye_indices = list(range(42, 48))  # 42-47
        
        left_eye = landmarks[left_eye_indices]
        right_eye = landmarks[right_eye_indices]
        
        return left_eye, right_eye
    
    def detect_blink(self, image: np.ndarray) -> Tuple[bool, dict]:
        """
        Detect eye blink in image (paper's native approach)
        Returns: (blink_detected, details)
        """
        eye_landmarks = self.get_eye_landmarks(image)
        
        if eye_landmarks is None:
            return False, {'left_ear': 0, 'right_ear': 0, 'avg_ear': 0}
        
        left_eye, right_eye = eye_landmarks
        
        # Calculate EAR for each eye
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Paper's approach: both eyes must blink together
        blink_detected = avg_ear < self.ear_threshold
        
        return blink_detected, {
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear
        }
    
    def detect_blink_proposed(self, image: np.ndarray) -> Tuple[bool, dict]:
        """
        Proposed blink detection (page 11): Each eye considered separately
        Blink detected if ANY eye's EAR < threshold
        """
        eye_landmarks = self.get_eye_landmarks(image)
        
        if eye_landmarks is None:
            return False, {'left_ear': 0, 'right_ear': 0, 'blink_type': 'none'}
        
        left_eye, right_eye = eye_landmarks
        
        # Calculate EAR for each eye
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        
        # Proposed approach: blink if any eye closes
        left_blink = left_ear < self.ear_threshold
        right_blink = right_ear < self.ear_threshold
        
        blink_detected = left_blink or right_blink
        
        # Determine blink type
        if left_blink and right_blink:
            blink_type = 'both'
        elif left_blink:
            blink_type = 'left'
        elif right_blink:
            blink_type = 'right'
        else:
            blink_type = 'none'
        
        return blink_detected, {
            'left_ear': left_ear,
            'right_ear': right_ear,
            'blink_type': blink_type
        }
    
    def detect_sequential_blinks(self, video_frames: List[np.ndarray], 
                                 method: str = 'proposed') -> Tuple[bool, int]:
        """
        Detect blinks in a sequence of frames
        Args:
            video_frames: List of consecutive frames
            method: 'native' or 'proposed'
        Returns: (blink_detected, num_blinks)
        """
        blink_count = 0
        blink_state = False  # True if currently in blink
        
        for frame in video_frames:
            if method == 'proposed':
                blink_detected, _ = self.detect_blink_proposed(frame)
            else:
                blink_detected, _ = self.detect_blink(frame)
            
            if blink_detected and not blink_state:
                # Start of a new blink
                blink_count += 1
                blink_state = True
            elif not blink_detected and blink_state:
                # End of blink
                blink_state = False
        
        return blink_count > 0, blink_count


class CNNLivenessDetector:
    """
    CNN-based liveness detector using adapted AlexNet (page 8-9)
    """
    
    def __init__(self):
        self.model = self._build_model()
        self.device = DEVICE
        
        if os.path.exists(LIVENESS_MODEL_PATH):
            self.load_model(LIVENESS_MODEL_PATH)
        else:
            print(f"Liveness model not found at {LIVENESS_MODEL_PATH}")
    
    def _build_model(self):
        """
        Build adapted AlexNet for liveness detection (page 9)
        Input: 64x64 RGB image
        """
        model = nn.Sequential(
            # Conv1: Input 64x64x3
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2), # Output: (64 - 5 + 2*2)/1 + 1 = 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (64 - 2)/2 + 1 = 32
            
            # Conv2: Input 32x32x64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Output: (32 - 3 + 2*1)/1 + 1 = 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (32 - 2)/2 + 1 = 16
            
            # Conv3: Input 16x16x128
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # Output: (16 - 3 + 2*1)/1 + 1 = 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (16 - 2)/2 + 1 = 8
            
            # Flatten: Input 8x8x256
            nn.Flatten(),
            
            # FC layers
            nn.Linear(256 * 8 * 8, 1024), # Adjusted input size for flatten
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 2) # Output: 2 classes (real vs fake)
        )
        
        return model
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for liveness detection"""
        # Resize to 64x64
        image_resized = cv2.resize(image, (64, 64))
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image_norm = image_rgb.astype(np.float32) / 255.0
        
        # Standard normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_norm = (image_norm - mean) / std
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
    
    def detect(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if image shows a live face
        Returns: (is_live, confidence)
        """
        self.model.eval()
        self.model.to(self.device)
        
        # Preprocess image
        image_tensor = self.preprocess(image).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Assuming index 0 is "real" and 1 is "fake"
            live_prob = probabilities[0, 0].item()
            is_live = live_prob > 0.5
            confidence = abs(live_prob - 0.5) * 2  # Normalize to [0, 1]
        
        return is_live, confidence
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': 'adapted_alexnet_64x64'
        }, path)
    
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()


class LivenessDetector:
    """Main liveness detector combining multiple methods"""
    
    def __init__(self):
        self.reflection_detector = ReflectionDetector()
        self.eye_blink_detector = EyeBlinkDetector()
        self.cnn_detector = CNNLivenessDetector()
    
    def detect(self, image: np.ndarray, use_cnn: bool = True) -> dict:
        """
        Comprehensive liveness detection
        Returns: Dictionary with detection results
        """
        results = {}
        
        # 1. Reflection-based detection (equation 7)
        reflection_live, reflection_coeff = self.reflection_detector.detect(image)
        results['reflection'] = {
            'is_live': reflection_live,
            'coefficient': reflection_coeff,
            'confidence': abs(reflection_coeff - LIVENESS_REFLECTION_THRESHOLD)
        }
        
        # 2. Eye blink detection (equation 8)
        blink_native, blink_details_native = self.eye_blink_detector.detect_blink(image)
        blink_proposed, blink_details_proposed = self.eye_blink_detector.detect_blink_proposed(image)
        
        results['blink'] = {
            'native': {
                'detected': blink_native,
                'details': blink_details_native
            },
            'proposed': {
                'detected': blink_proposed,
                'details': blink_details_proposed
            }
        }
        
        # 3. CNN-based detection (if enabled)
        if use_cnn and hasattr(self.cnn_detector, 'model'):
            cnn_live, cnn_confidence = self.cnn_detector.detect(image)
            results['cnn'] = {
                'is_live': cnn_live,
                'confidence': cnn_confidence
            }
        else:
            results['cnn'] = {
                'is_live': True,  # Assume live if CNN not available
                'confidence': 0.5
            }
        
        # 4. Combined decision
        # Weighted combination of all methods
        weights = {
            'reflection': 0.3,
            'blink': 0.3,
            'cnn': 0.4
        }
        
        live_score = 0
        total_weight = 0
        
        # Reflection score
        live_score += weights['reflection'] * (1.0 if reflection_live else 0.0)
        total_weight += weights['reflection']
        
        # Blink score (use proposed method)
        blink_score = 1.0 if blink_proposed else 0.0
        live_score += weights['blink'] * blink_score
        total_weight += weights['blink']
        
        # CNN score
        cnn_score = 1.0 if results['cnn']['is_live'] else 0.0
        live_score += weights['cnn'] * cnn_score
        total_weight += weights['cnn']
        
        # Normalize score
        if total_weight > 0:
            live_score /= total_weight
        
        results['combined'] = {
            'score': live_score,
            'is_live': live_score > 0.5,
            'confidence': abs(live_score - 0.5) * 2
        }
        
        return results
    
    def get_authentication_vector(self, image: np.ndarray) -> list:
        """
        Get authentication vector A = [a1, a2] as per paper (page 3)
        a1: Reflection-based liveness
        a2: Eye blink detection (proposed method)
        """
        # Get detection results
        results = self.detect(image, use_cnn=False)
        
        # a1: Reflection coefficient > threshold
        a1 = 1 if results['reflection']['is_live'] else 0
        
        # a2: Eye blink detected (proposed method)
        a2 = 1 if results['blink']['proposed']['detected'] else 0
        
        return [a1, a2]
    
    def process_video(self, video_path: str, frame_interval: int = 5) -> dict:
        """
        Process video for liveness detection
        Analyzes frames at intervals to detect blinks and other liveness cues
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = 0
        blink_count = 0
        live_frames = 0
        total_frames = 0
        
        frames_to_analyze = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Analyze every nth frame
            if frame_count % frame_interval == 0:
                # Detect blink in this frame
                blink_detected, _ = self.eye_blink_detector.detect_blink_proposed(frame)
                if blink_detected:
                    blink_count += 1
                
                # Store frame for later analysis if needed
                frames_to_analyze.append(frame.copy())
                total_frames += 1
        
        cap.release()
        
        # Calculate blink rate
        blink_rate = blink_count / total_frames if total_frames > 0 else 0
        
        # Analyze a sample of frames for overall liveness
        sample_frames = frames_to_analyze[:min(10, len(frames_to_analyze))]
        live_samples = 0
        
        for frame in sample_frames:
            results = self.detect(frame)
            if results['combined']['is_live']:
                live_samples += 1
        
        live_ratio = live_samples / len(sample_frames) if sample_frames else 0
        
        return {
            'total_frames': frame_count,
            'analyzed_frames': total_frames,
            'blink_count': blink_count,
            'blink_rate': blink_rate,
            'live_ratio': live_ratio,
            'is_live': live_ratio > 0.5 and blink_rate > 0.01
        }


# Global liveness detector instance
liveness_detector = LivenessDetector()
