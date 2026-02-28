"""
Face detection using MTCNN as per paper (page 3)
"""
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from config import FACE_DETECTION_CONFIDENCE, DEVICE
from utils import get_timestamp

class FaceDetector:
    """MTCNN-based face detector with alignment"""
    
    def __init__(self, min_face_size=20, keep_all=False):
        """
        Initialize MTCNN detector
        Args:
            min_face_size: Minimum face size to detect
            keep_all: Whether to keep all detected faces or just the largest
        """
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=keep_all,
            device=DEVICE
        )
        self.min_confidence = FACE_DETECTION_CONFIDENCE
        self.keep_all = keep_all
    
    def detect_faces(self, image):
        """
        Detect faces in an image
        Returns:
            List of tuples (face_image, bounding_box, confidence, landmarks)
        """
        # Convert BGR to RGB for MTCNN
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Detect faces
        boxes, probs, landmarks = self.mtcnn.detect(image_pil, landmarks=True)
        
        detected_faces = []
        
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob >= self.min_confidence:
                    # Convert box coordinates to integers
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Ensure box is within image bounds
                    height, width = image.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    
                    # Extract face region
                    face_region = image[y1:y2, x1:x2]
                    
                    if face_region.size == 0:
                        continue
                    
                    # Get landmarks for this face
                    face_landmarks = None
                    if landmarks is not None and i < len(landmarks):
                        face_landmarks = landmarks[i]
                    
                    detected_faces.append({
                        'face': face_region,
                        'box': (x1, y1, x2, y2),
                        'confidence': prob,
                        'landmarks': face_landmarks
                    })
        
        # Sort by confidence and keep only the best if keep_all is False
        if not self.keep_all and detected_faces:
            detected_faces = [max(detected_faces, key=lambda x: x['confidence'])]
        
        return detected_faces
    
    def detect_and_align(self, image):
        """
        Detect and align faces using MTCNN's built-in alignment
        Returns aligned face images
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Detect and align
        faces = self.mtcnn(image_pil, return_prob=True)
        
        if faces[0] is None:
            return []
        
        if self.keep_all:
            aligned_faces = []
            for face_tensor, prob in zip(*faces):
                if prob >= self.min_confidence:
                    # Convert tensor to numpy
                    face_np = face_tensor.permute(1, 2, 0).cpu().numpy() * 255
                    face_np = face_np.astype(np.uint8)
                    aligned_faces.append(face_np)
            return aligned_faces
        else:
            # Return only the face with highest confidence
            best_idx = torch.argmax(faces[1]).item()
            if faces[1][best_idx] >= self.min_confidence:
                face_np = faces[0][best_idx].permute(1, 2, 0).cpu().numpy() * 255
                return [face_np.astype(np.uint8)]
            else:
                return []
    
    def extract_face_regions(self, image, boxes):
        """Extract face regions given bounding boxes"""
        faces = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            height, width = image.shape[:2]
            
            # Ensure box is within bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            face = image[y1:y2, x1:x2]
            if face.size > 0:
                faces.append(face)
        
        return faces
    
    def draw_detections(self, image, detections):
        """Draw face detections on image"""
        result = image.copy()
        
        for detection in detections:
            box = detection['box']
            confidence = detection['confidence']
            landmarks = detection['landmarks']
            
            # Draw bounding box
            x1, y1, x2, y2 = box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f"Face: {confidence:.2f}"
            cv2.putText(result, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw landmarks
            if landmarks is not None:
                for (x, y) in landmarks:
                    cv2.circle(result, (int(x), int(y)), 2, (0, 0, 255), -1)
        
        return result
    
    def batch_detect(self, image_list):
        """Detect faces in batch of images"""
        all_detections = []
        
        for image in image_list:
            detections = self.detect_faces(image)
            all_detections.append(detections)
        
        return all_detections


# Global detector instance
face_detector = FaceDetector(keep_all=False)