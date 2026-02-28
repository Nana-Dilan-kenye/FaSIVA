#!/usr/bin/env python3
"""
Main FaSIVA pipeline implementation (page 4, Figure 1)
"""
import os
import cv2
import numpy as np
import json
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import torch
from datetime import datetime, timedelta

from config import *
from utils import (
    load_image, save_image, check_resolution, 
    get_resolution, euclidean_distance, setup_logging,
    calculate_metrics
)
from face_detection import face_detector
from super_resolution import apply_super_resolution, get_fsrcnn_model
from feature_extraction import feature_extractor
from liveness_detection import liveness_detector
from database import FaceDatabase

logger = setup_logging()

class FaSIVA:
    """
    Main FaSIVA (Facial Signature for Identification, Verification and Authentication)
    Implementation based on the research paper
    """
    
    def __init__(self, database_path: str = None):
        """
        Initialize FaSIVA system
        Args:
            database_path: Path to face database (creates new if None)
        """
        logger.info("Initializing FaSIVA system...")
        
        # Initialize components
        self.face_detector = face_detector
        self.feature_extractor = feature_extractor
        self.liveness_detector = liveness_detector
        
        # Try to load FSRCNN model
        try:
            self.fsrcnn_model = get_fsrcnn_model()
            logger.info("FSRCNN model loaded successfully")
        except Exception as e:
            logger.warning(f"FSRCNN model not available: {e}")
            self.fsrcnn_model = None
        
        # Initialize database
        self.database = FaceDatabase(database_path)
        
        # Statistics
        self.stats = {
            'processed_images': 0,
            'successful_identifications': 0,
            'failed_identifications': 0,
            'spoofing_attempts': 0
        }
        
        logger.info("FaSIVA system initialized successfully")
    
    def process_image(self, image_path: str, save_signature: bool = False) -> Dict[str, Any]:
        """
        Process a single image through the complete FaSIVA pipeline
        Follows the flowchart in Figure 1 of the paper
        """
        logger.info(f"Processing image: {image_path}")
        
        try:
            # Step 0: Load image
            image = load_image(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Step 1: Face detection using MTCNN (page 3)
            logger.debug("Step 1: Face detection")
            detections = self.face_detector.detect_faces(image)
            
            if not detections:
                logger.warning("No face detected in image")
                return {
                    'success': False,
                    'error': 'No face detected',
                    'image_path': image_path
                }
            
            # Use the first (or only) detected face
            face_data = detections[0]
            face_image = face_data['face']
            
            # Check if face_image is valid
            if face_image is None:
                raise ValueError("MTCNN returned None for face image")
                
            logger.info(f"Face detected: {face_image.shape}")
            
            # Convert face_image to proper format if needed
            if isinstance(face_image, list):
                face_image = np.array(face_image)
            
            # Ensure it's a numpy array
            if not isinstance(face_image, np.ndarray):
                raise TypeError(f"Face image is not a numpy array: {type(face_image)}")
            
            # Step 2: Check resolution and apply super-resolution if needed (page 4)
            logger.debug("Step 2: Resolution check and super-resolution")
            resolution_before = get_resolution(face_image)
            
            if not check_resolution(face_image):
                logger.info(f"Low resolution detected: {resolution_before}. Applying super-resolution...")
                if self.fsrcnn_model is not None:
                    try:
                        # Make sure face_image is in correct format for FSRCNN
                        if len(face_image.shape) == 2:
                            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
                        elif face_image.shape[2] == 4:
                            face_image = face_image[:, :, :3]
                            
                        face_image = apply_super_resolution(face_image, self.fsrcnn_model)
                        resolution_after = get_resolution(face_image)
                        logger.info(f"Super-resolution applied: {resolution_before} -> {resolution_after}")
                    except Exception as e:
                        logger.error(f"Super-resolution failed: {e}")
                        # Continue with original image
                else:
                    logger.warning("FSRCNN model not available, skipping super-resolution")
            else:
                logger.info(f"Resolution OK: {resolution_before}")
            
            # Step 3: Extract feature vectors (page 3-4)
            logger.debug("Step 3: Feature extraction")
            
            # Extract F vector (identification) - ResNet-50, 2062 dimensions
            f_vector = self.feature_extractor.extract_f_vector(face_image)
            
            # Extract E vector (verification) - FaceNet, 128 dimensions
            e_vector = self.feature_extractor.extract_e_vector(face_image)
            
            # Extract A vector (authentication) - liveness detection
            a_vector = self.liveness_detector.get_authentication_vector(face_image)
            
            # Step 4: Create complete FaSIVA signature (equation 2, page 3)
            signature = {
                'R': get_resolution(face_image),  # Resolution
                'F': f_vector,                     # Identification vector (2062D)
                'E': e_vector,                     # Verification vector (128D)
                'A': a_vector                      # Authentication vector [a1, a2]
            }
            
            logger.info(f"FaSIVA signature created: R={signature['R']}, A={signature['A']}")
            
            # Step 5: Save signature if requested
            if save_signature:
                signature_path = f"signature_{os.path.basename(image_path).split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                from utils import save_signature
                save_signature(signature, signature_path)
                logger.info(f"Signature saved to {signature_path}")
            
            # Update statistics
            self.stats['processed_images'] += 1
            
            return {
                'success': True,
                'image_path': image_path,
                'face_detected': True,
                'face_image': face_image,
                'signature': signature,
                'detection_data': face_data
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path
            }
    
    def identify_person(self, image_path: str, threshold: float = None) -> Dict[str, Any]:
        """
        Identify a person from an image (page 4, step 2)
        Returns identification results
        """
        threshold = threshold or IDENTIFICATION_THRESHOLD
        
        logger.info(f"Identifying person from image: {image_path}")
        
        # Process image to get signature
        result = self.process_image(image_path)
        
        if not result['success']:
            return result
        
        signature = result['signature']
        f_vector = signature['F']
        
        # Step 2: Identification using F vector
        logger.debug("Step 2: Person identification")
        person_id, distance = self.database.find_person_by_f_vector(f_vector, threshold)
        
        if person_id is not None:
            logger.info(f"Person identified: ID={person_id}, distance={distance:.4f}")
            
            # Get person details from database
            cursor = self.database.conn.cursor()
            cursor.execute("SELECT name FROM persons WHERE id = ?", (person_id,))
            row = cursor.fetchone()
            person_name = row[0] if row else "Unknown"
            
            result['identification'] = {
                'success': True,
                'person_id': person_id,
                'person_name': person_name,
                'distance': distance,
                'threshold': threshold
            }
            
            self.stats['successful_identifications'] += 1
        else:
            logger.info(f"No matching person found (min distance: {distance:.4f})")
            result['identification'] = {
                'success': False,
                'distance': distance,
                'threshold': threshold
            }
            self.stats['failed_identifications'] += 1
        
        return result
    
    def verify_person(self, image_path: str, claimed_person_id: int, 
                     threshold: float = None) -> Dict[str, Any]:
        """
        Verify a person's identity (page 4, step 3)
        Returns verification results
        """
        threshold = threshold or VERIFICATION_THRESHOLD
        
        logger.info(f"Verifying person ID {claimed_person_id} from image: {image_path}")
        
        # Process image to get signature
        result = self.process_image(image_path)
        
        if not result['success']:
            return result
        
        signature = result['signature']
        e_vector = signature['E']
        
        # Step 3: Verification using E vector
        logger.debug("Step 3: Person verification")
        verified, distance = self.database.verify_person(claimed_person_id, e_vector, threshold)
        
        # Get person details
        cursor = self.database.conn.cursor()
        cursor.execute("SELECT name FROM persons WHERE id = ?", (claimed_person_id,))
        row = cursor.fetchone()
        person_name = row[0] if row else "Unknown"
        
        result['verification'] = {
            'success': verified,
            'claimed_person_id': claimed_person_id,
            'claimed_person_name': person_name,
            'distance': distance,
            'threshold': threshold,
            'verified': verified
        }
        
        if verified:
            logger.info(f"Person verified: {person_name} (ID: {claimed_person_id}), distance={distance:.4f}")
        else:
            logger.warning(f"Person NOT verified: {person_name} (ID: {claimed_person_id}), distance={distance:.4f}")
        
        return result
    
    def authenticate_person(self, image_path: str, claimed_person_id: int = None) -> Dict[str, Any]:
        """
        Complete authentication: identification + verification + liveness (page 4, step 4)
        Returns authentication results
        """
        logger.info(f"Authenticating person from image: {image_path}")
        
        # Step 1: Process image to get signature
        result = self.process_image(image_path)
        
        if not result['success']:
            return result
        
        signature = result['signature']
        f_vector = signature['F']
        e_vector = signature['E']
        a_vector = signature['A']
        
        # Step 2: Identification (if person_id not provided)
        if claimed_person_id is None:
            logger.debug("Step 2a: Person identification")
            person_id, id_distance = self.database.find_person_by_f_vector(
                f_vector, IDENTIFICATION_THRESHOLD
            )
            
            if person_id is None:
                logger.warning("Authentication failed: Person not identified")
                result['authentication'] = {
                    'success': False,
                    'stage_failed': 'identification',
                    'reason': 'Person not found in database'
                }
                return result
            
            claimed_person_id = person_id
            identification_success = True
        else:
            identification_success = True
            id_distance = 0  # Not computed when person_id is provided
        
        # Step 3: Verification
        logger.debug("Step 3: Person verification")
        verified, ver_distance = self.database.verify_person(
            claimed_person_id, e_vector, VERIFICATION_THRESHOLD
        )
        
        if not verified:
            logger.warning("Authentication failed: Verification failed")
            result['authentication'] = {
                'success': False,
                'stage_failed': 'verification',
                'reason': 'Face verification failed'
            }
            return result
        
        # Step 4: Liveness check (authentication vector A)
        logger.debug("Step 4: Liveness check")
        liveness_check = all(a_vector)  # Both a1 and a2 must be 1
        
        if not liveness_check:
            logger.warning("Authentication failed: Liveness check failed")
            self.stats['spoofing_attempts'] += 1
            result['authentication'] = {
                'success': False,
                'stage_failed': 'liveness',
                'reason': 'Liveness detection failed',
                'a_vector': a_vector
            }
            
            # Log suspicious access attempt
            self.database.log_access(
                claimed_person_id, 'suspicious', 
                max(0, 1 - ver_distance), False
            )
            return result
        
        # Authentication successful!
        logger.info(f"Authentication SUCCESSFUL for person ID {claimed_person_id}")
        
        # Get person details
        cursor = self.database.conn.cursor()
        cursor.execute("SELECT name FROM persons WHERE id = ?", (claimed_person_id,))
        row = cursor.fetchone()
        person_name = row[0] if row else "Unknown"
        
        # Calculate overall confidence
        confidence = max(0, 1 - ver_distance)  # Higher distance = lower confidence
       
        # Log successful access
        self.database.log_access(
            claimed_person_id, 'granted', confidence, True
        )
        
        result['authentication'] = {
            'success': True,
            'person_id': claimed_person_id,
            'person_name': person_name,
            'identification_success': identification_success,
            'identification_distance': id_distance,
            'verification_success': verified,
            'verification_distance': ver_distance,
            'liveness_success': liveness_check,
            'a_vector': a_vector,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def register_person(self, image_paths: list, person_name: str, 
                       num_samples: int = 10) -> Dict[str, Any]:
        """
        Register a new person in the database
        Args:
            image_paths: List of image paths for the person
            person_name: Name of the person
            num_samples: Number of samples to use (use first n images)
        """
        logger.info(f"Registering new person: {person_name}")
        
        # Create new person entry
        person_id = self.database.add_person(person_name)
        logger.info(f"Created person entry with ID: {person_id}")
        
        # Process and add signatures
        successful_registrations = 0
        signatures = []
        
        for i, image_path in enumerate(image_paths[:num_samples]):
            try:
                # Process image
                result = self.process_image(image_path)
                
                if result['success']:
                    signature = result['signature']
                    
                    # Add to database
                    self.database.add_signature(person_id, signature)
                    signatures.append(signature)
                    successful_registrations += 1
                    
                    logger.info(f"Added signature {i+1}/{min(num_samples, len(image_paths))}")
                else:
                    logger.warning(f"Failed to process image {image_path}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
        
        # Calculate average feature vectors if we have signatures
        if signatures:
            avg_f_vector = np.mean([sig['F'] for sig in signatures], axis=0)
            avg_e_vector = np.mean([sig['E'] for sig in signatures], axis=0)
            avg_a_vector = [int(any(sig['A'][0] for sig in signatures)), 
                          int(any(sig['A'][1] for sig in signatures))]
            
            avg_signature = {
                'R': signatures[0]['R'],  # Use first signature's resolution
                'F': avg_f_vector,
                'E': avg_e_vector,
                'A': avg_a_vector
            }
            
            # Add average signature as well
            try:
                self.database.add_signature(person_id, avg_signature)
                logger.info("Added average signature to database")
            except Exception as e:
                logger.error(f"Failed to add average signature: {e}")
        
        return {
            'success': successful_registrations > 0,
            'person_id': person_id,
            'person_name': person_name,
            'signatures_added': successful_registrations,
            'total_samples': min(num_samples, len(image_paths))
        }

    def batch_process(self, image_paths: list, mode: str = 'identify') -> list:
        """
        Process multiple images in batch
        Args:
            image_paths: List of image paths
            mode: 'identify', 'verify', or 'authenticate'
        """
        results = []
        
        logger.info(f"Batch processing {len(image_paths)} images in {mode} mode")
        
        for image_path in image_paths:
            try:
                if mode == 'identify':
                    result = self.identify_person(image_path)
                elif mode == 'authenticate':
                    result = self.authenticate_person(image_path)
                else:
                    result = self.process_image(image_path)
                
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'success': False,
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def evaluate_system(self, test_dataset: list, ground_truth: dict) -> Dict[str, Any]:
        """
        Evaluate system performance on test dataset
        Args:
            test_dataset: List of (image_path, person_id) tuples
            ground_truth: Dictionary mapping image_path to true person_id
        Returns evaluation metrics
        """
        logger.info(f"Evaluating system on {len(test_dataset)} test samples")
        
        tp = fp = tn = fn = 0
        
        for image_path, true_person_id in test_dataset:
            try:
                # Authenticate person
                result = self.authenticate_person(image_path)
                
                if result['success'] and 'authentication' in result:
                    auth_result = result['authentication']
                    
                    if auth_result['success']:
                        # Authentication successful
                        predicted_person_id = auth_result['person_id']
                        
                        if predicted_person_id == true_person_id:
                            tp += 1  # Correctly authenticated
                        else:
                            fp += 1  # Wrong person authenticated
                    else:
                        # Authentication failed
                        if true_person_id is None:
                            tn += 1  # Correctly rejected (no person should be authenticated)
                        else:
                            fn += 1  # Incorrectly rejected (should have authenticated)
                else:
                    # Processing failed
                    logger.warning(f"Failed to process {image_path}")
            
            except Exception as e:
                logger.error(f"Error evaluating {image_path}: {e}")
        
        # Calculate metrics (page 11, equation 10)
        metrics = calculate_metrics(tp, fp, tn, fn)
        
        logger.info(f"Evaluation complete: ACC={metrics['ACC']:.4f}, FAR={metrics['FAR']:.4f}, FRR={metrics['FRR']:.4f}")
        
        return {
            'metrics': metrics,
            'counts': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
            'total_samples': len(test_dataset)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        db_stats = self.database.get_statistics()
        
        return {
            'system_stats': self.stats,
            'database_stats': db_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def close(self):
        """Close system and clean up resources"""
        logger.info("Closing FaSIVA system...")
        if hasattr(self.database, 'close'):
            self.database.close()
        logger.info("FaSIVA system closed")


def main():
    """Main function to demonstrate FaSIVA usage"""
    print("=" * 60)
    print("FaSIVA: Facial Signature for Identification, Verification and Authentication")
    print("=" * 60)
    
    # Initialize FaSIVA system
    fasiva = FaSIVA()
    
    try:
        # Example 1: Register a new person
        print("\n1. Registering a new person...")
        
        # You need to replace these with actual image paths
        sample_images = [
            "dataset/test/Andrej/1.jpeg",
            "dataset/test/Andrej/2.jpeg",
            "dataset/test/Andrej/3.jpeg",
            "dataset/test/Andrej/4.png",
            "dataset/test/Andrej/5_.jpeg",
            "dataset/test/Andrej/6.jpeg",
            "dataset/test/Andrej/7.jpeg",
            "dataset/test/Andrej/8.jpeg",
            "dataset/test/Andrej/9.jpeg",
            "dataset/test/Andrej/10.jpeg",
            "dataset/test/Andrej/11.jpg",
        ]
        
        # Check if sample images exist
        existing_images = [img for img in sample_images if os.path.exists(img)]
        
        if existing_images:
            registration_result = fasiva.register_person(
                existing_images, 
                person_name="Andrej Karpathy",
                num_samples=max(3, len(existing_images))
            )
            
            if registration_result['success']:
                person_id = registration_result['person_id']
                print(f"✓ Person registered successfully! ID: {person_id}")
            else:
                print("✗ Registration failed. Using demo mode...")
                person_id = 1  # Demo ID
        else:
            print("✗ Sample images not found. Using demo mode...")
            person_id = 1  # Demo ID
        
        # Example 2: Identify a person
        print("\n2. Identifying a person...")
        test_image = "dataset/test/Andrej/13.png"  # Replace with actual test image path
        
        if os.path.exists(test_image):
            identification_result = fasiva.identify_person(test_image)
            
            if identification_result['success']:
                if identification_result.get('identification', {}).get('success'):
                    identified_person = identification_result['identification']['person_name']
                    distance = identification_result['identification']['distance']
                    print(f"✓ Identified as: {identified_person} (distance: {distance:.4f})")
                else:
                    print("✗ Person not identified in database")
            else:
                print(f"✗ Identification failed: {identification_result.get('error', 'Unknown error')}")
        else:
            print(f"✗ Test image not found: {test_image}")
        
        # Example 3: Authenticate a person
        print("\n3. Authenticating a person...")
        
        if os.path.exists(test_image):
            authentication_result = fasiva.authenticate_person(test_image, person_id)
            
            if authentication_result['success']:
                auth = authentication_result.get('authentication', {})
                if auth.get('success'):
                    print(f"✓ AUTHENTICATION SUCCESSFUL!")
                    print(f"   Person: {auth.get('person_name', 'Unknown')} (ID: {auth.get('person_id', 'Unknown')})")
                    print(f"   Confidence: {auth.get('confidence', 0):.2%}")
                    print(f"   Liveness: {auth.get('a_vector', [])}")
                else:
                    print(f"✗ Authentication failed at stage: {auth.get('stage_failed', 'Unknown')}")
                    print(f"   Reason: {auth.get('reason', 'Unknown')}")
            else:
                print(f"✗ Authentication process failed: {authentication_result.get('error', 'Unknown error')}")
        else:
            print(f"✗ Test image not found: {test_image}")
        
        # Example 4: Get system statistics
        print("\n4. System Statistics:")
        stats = fasiva.get_statistics()
        print(f"   Processed images: {stats['system_stats']['processed_images']}")
        print(f"   Successful identifications: {stats['system_stats']['successful_identifications']}")
        print(f"   Spoofing attempts detected: {stats['system_stats']['spoofing_attempts']}")
        
        db_stats = stats['database_stats']
        print(f"   Persons in database: {db_stats.get('total_persons', 0)}")
        print(f"   Total signatures: {db_stats.get('total_signatures', 0)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close system
        fasiva.close()
        print("\nFaSIVA system closed.")


if __name__ == "__main__":
    # Create necessary directories
    from utils import create_directory_structure
    create_directory_structure()
    
    # Run main function
    main()