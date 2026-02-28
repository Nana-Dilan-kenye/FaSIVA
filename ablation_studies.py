"""
FaSIVA Ablation Studies
Test contribution of each component to overall performance
"""
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import json

from config import *
from main import FaSIVA
from feature_extraction import feature_extractor
from utils import load_image, euclidean_distance


class AblationStudies:
    """Perform ablation studies on FaSIVA components"""
    
    def __init__(self):
        self.fasiva = FaSIVA()
        self.results = {}
    
    def test_without_super_resolution(self, test_pairs: List[Tuple[str, str, bool]]) -> Dict:
        """
        Test identification without super-resolution enhancement
        
        Args:
            test_pairs: List of (img1_path, img2_path, is_same_identity) tuples
        """
        print(f"\n{'=' * 60}")
        print("ABLATION: Without Super-Resolution Module")
        print(f"{'=' * 60}")
        
        results = {
            'correct_matches': 0,
            'total_tests': len(test_pairs),
            'accuracy': 0.0,
            'avg_distance_same': 0.0,
            'avg_distance_different': 0.0
        }
        
        distances_same = []
        distances_different = []
        
        for img1_path, img2_path, is_same in tqdm(test_pairs, desc="Testing without SR"):
            try:
                if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                    continue
                
                img1 = load_image(img1_path)
                img2 = load_image(img2_path)
                
                # Process without super-resolution
                faces1 = self.fasiva.face_detector.detect_faces(img1)
                faces2 = self.fasiva.face_detector.detect_faces(img2)
                
                if faces1 and faces2:
                    f1_vec, _ = feature_extractor.extract_signature_features(faces1[0]['face'])
                    f2_vec, _ = feature_extractor.extract_signature_features(faces2[0]['face'])
                    
                    distance = euclidean_distance(f1_vec, f2_vec)
                    
                    if is_same:
                        distances_same.append(distance)
                        if distance <= IDENTIFICATION_THRESHOLD:
                            results['correct_matches'] += 1
                    else:
                        distances_different.append(distance)
                        if distance > IDENTIFICATION_THRESHOLD:
                            results['correct_matches'] += 1
            except:
                continue
        
        if results['total_tests'] > 0:
            results['accuracy'] = results['correct_matches'] / results['total_tests']
        
        if distances_same:
            results['avg_distance_same'] = float(np.mean(distances_same))
        if distances_different:
            results['avg_distance_different'] = float(np.mean(distances_different))
        
        print(f"✓ Non-SR Accuracy: {results['accuracy']:.4f}")
        print(f"✓ Avg distance (same identity): {results['avg_distance_same']:.4f}")
        print(f"✓ Avg distance (different identity): {results['avg_distance_different']:.4f}")
        
        return results
    
    def test_without_liveness_detection(self, test_images: List[str], 
                                       fake_images: List[str]) -> Dict:
        """
        Test system without liveness detection (vulnerable to spoofing)
        
        Args:
            test_images: Real face images
            fake_images: Spoofed/fake face images
        """
        print(f"\n{'=' * 60}")
        print("ABLATION: Without Liveness Detection")
        print(f"{'=' * 60}")
        
        results = {
            'real_accepted': 0,
            'total_real': len(test_images),
            'fake_accepted': 0,
            'total_fake': len(fake_images),
            'spoofing_success_rate': 0.0
        }
        
        # Real faces should be accepted
        for img_path in tqdm(test_images[:50], desc="Testing real (no liveness)"):
            try:
                if not os.path.exists(img_path):
                    continue
                
                image = load_image(img_path)
                faces = self.fasiva.face_detector.detect_faces(image)
                
                if faces:
                    # Would be accepted without liveness check
                    results['real_accepted'] += 1
            except:
                continue
        
        # Fake faces would also be accepted (VULNERABILITY)
        for img_path in tqdm(fake_images[:50], desc="Testing fakes (no liveness)"):
            try:
                if not os.path.exists(img_path):
                    continue
                
                image = load_image(img_path)
                faces = self.fasiva.face_detector.detect_faces(image)
                
                if faces:
                    # Would be accepted without liveness check
                    results['fake_accepted'] += 1
            except:
                continue
        
        if results['total_real'] > 0:
            results['real_acceptance_rate'] = results['real_accepted'] / results['total_real']
        
        if results['total_fake'] > 0:
            results['spoofing_success_rate'] = results['fake_accepted'] / results['total_fake']
        
        print(f"✓ Real face acceptance rate: {results['real_acceptance_rate']:.4f}")
        print(f"✓ Spoofing success rate WITHOUT liveness: {results['spoofing_success_rate']:.4f}")
        print(f"  (This shows the vulnerability without liveness detection!)")
        
        return results
    
    def test_without_verification(self, test_pairs: List[Tuple[str, str, bool]]) -> Dict:
        """
        Test identification without verification module
        (Uses only F vector, not E vector)
        """
        print(f"\n{'=' * 60}")
        print("ABLATION: Without Verification Module (E Vector)")
        print(f"{'=' * 60}")
        
        results = {
            'correct_matches': 0,
            'total_tests': len(test_pairs),
            'accuracy': 0.0
        }
        
        for img1_path, img2_path, is_same in tqdm(test_pairs, desc="Testing without verification"):
            try:
                if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                    continue
                
                img1 = load_image(img1_path)
                img2 = load_image(img2_path)
                
                faces1 = self.fasiva.face_detector.detect_faces(img1)
                faces2 = self.fasiva.face_detector.detect_faces(img2)
                
                if faces1 and faces2:
                    # Use only F vector (identification), ignore E vector (verification)
                    f1_vec, _ = feature_extractor.extract_signature_features(faces1[0]['face'])
                    f2_vec, _ = feature_extractor.extract_signature_features(faces2[0]['face'])
                    
                    distance = euclidean_distance(f1_vec, f2_vec)
                    
                    if is_same:
                        if distance <= IDENTIFICATION_THRESHOLD:
                            results['correct_matches'] += 1
                    else:
                        if distance > IDENTIFICATION_THRESHOLD:
                            results['correct_matches'] += 1
            except:
                continue
        
        if results['total_tests'] > 0:
            results['accuracy'] = results['correct_matches'] / results['total_tests']
        
        print(f"✓ Accuracy without verification module: {results['accuracy']:.4f}")
        
        return results
    
    def test_f_vector_only(self, test_pairs: List[Tuple[str, str, bool]]) -> Dict:
        """Test with F vector only (ResNet-50)"""
        
        print(f"\n{'=' * 60}")
        print("ABLATION: F Vector Only (ResNet-50 Identification)")
        print(f"{'=' * 60}")
        
        results = {
            'correct_matches': 0,
            'total_tests': len(test_pairs),
            'accuracy': 0.0
        }
        
        for img1_path, img2_path, is_same in tqdm(test_pairs, desc="Testing F-vector only"):
            try:
                if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                    continue
                
                img1 = load_image(img1_path)
                img2 = load_image(img2_path)
                
                faces1 = self.fasiva.face_detector.detect_faces(img1)
                faces2 = self.fasiva.face_detector.detect_faces(img2)
                
                if faces1 and faces2:
                    # F vector (ResNet-50)
                    f1_vec, _ = feature_extractor.extract_signature_features(faces1[0]['face'])
                    f2_vec, _ = feature_extractor.extract_signature_features(faces2[0]['face'])
                    
                    distance = euclidean_distance(f1_vec, f2_vec)
                    
                    if is_same:
                        if distance <= IDENTIFICATION_THRESHOLD:
                            results['correct_matches'] += 1
                    else:
                        if distance > IDENTIFICATION_THRESHOLD:
                            results['correct_matches'] += 1
            except:
                continue
        
        if results['total_tests'] > 0:
            results['accuracy'] = results['correct_matches'] / results['total_tests']
        
        print(f"✓ F-vector only accuracy: {results['accuracy']:.4f}")
        
        return results
    
    def test_e_vector_only(self, test_pairs: List[Tuple[str, str, bool]]) -> Dict:
        """Test with E vector only (FaceNet)"""
        
        print(f"\n{'=' * 60}")
        print("ABLATION: E Vector Only (FaceNet Verification)")
        print(f"{'=' * 60}")
        
        results = {
            'correct_matches': 0,
            'total_tests': len(test_pairs),
            'accuracy': 0.0
        }
        
        FACENET_THRESHOLD = 0.6  # Typical FaceNet threshold
        
        for img1_path, img2_path, is_same in tqdm(test_pairs, desc="Testing E-vector only"):
            try:
                if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                    continue
                
                img1 = load_image(img1_path)
                img2 = load_image(img2_path)
                
                faces1 = self.fasiva.face_detector.detect_faces(img1)
                faces2 = self.fasiva.face_detector.detect_faces(img2)
                
                if faces1 and faces2:
                    # E vector (FaceNet)
                    _, e1_vec = feature_extractor.extract_signature_features(faces1[0]['face'])
                    _, e2_vec = feature_extractor.extract_signature_features(faces2[0]['face'])
                    
                    distance = euclidean_distance(e1_vec, e2_vec)
                    
                    if is_same:
                        if distance <= FACENET_THRESHOLD:
                            results['correct_matches'] += 1
                    else:
                        if distance > FACENET_THRESHOLD:
                            results['correct_matches'] += 1
            except:
                continue
        
        if results['total_tests'] > 0:
            results['accuracy'] = results['correct_matches'] / results['total_tests']
        
        print(f"✓ E-vector only accuracy: {results['accuracy']:.4f}")
        
        return results
    
    def generate_ablation_report(self, test_pairs: List[Tuple[str, str, bool]],
                                real_images: List[str],
                                fake_images: List[str]) -> Dict:
        """Generate comprehensive ablation study report"""
        
        print("\n" + "=" * 70)
        print("FASIVA ABLATION STUDIES")
        print("=" * 70)
        
        report = {
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'ablation_tests': {}
        }
        
        # Component ablations
        report['ablation_tests']['without_super_resolution'] = self.test_without_super_resolution(test_pairs)
        report['ablation_tests']['without_liveness_detection'] = self.test_without_liveness_detection(
            real_images, fake_images
        )
        report['ablation_tests']['without_verification'] = self.test_without_verification(test_pairs)
        
        # Vector-specific ablations
        report['ablation_tests']['f_vector_only'] = self.test_f_vector_only(test_pairs)
        report['ablation_tests']['e_vector_only'] = self.test_e_vector_only(test_pairs)
        
        return report
    
    def save_ablation_report(self, report: Dict, output_path: str = "ablation_report.json"):
        """Save ablation study report"""
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Ablation report saved to {output_path}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("ABLATION STUDY SUMMARY")
        print("=" * 70)
        
        for test_name, results in report['ablation_tests'].items():
            print(f"\n{test_name}:")
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        return report


def run_ablation_studies():
    """Run all ablation studies on current datasets"""
    
    ablation = AblationStudies()
    
    # Prepare test data
    lfw_dir = os.path.join(DATASETS_DIR, 'lfw-deepfunneled', 'lfw-deepfunneled')
    nuaa_real = os.path.join(DATASETS_DIR, 'CASIA2', 'ClientRaw')
    nuaa_fake = os.path.join(DATASETS_DIR, 'CASIA2', 'ImposterRaw')
    
    # Collect test pairs from LFW
    test_pairs = []
    real_images = []
    fake_images = []
    
    if os.path.exists(lfw_dir):
        for person_name in os.listdir(lfw_dir)[:50]:  # Use first 50 identities
            person_dir = os.path.join(lfw_dir, person_name)
            if os.path.isdir(person_dir):
                images = [os.path.join(person_dir, f) 
                         for f in os.listdir(person_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Matched pairs
                if len(images) >= 2:
                    test_pairs.append((images[0], images[1], True))
                
                # Real images for spoofing test
                real_images.extend(images[:2])
    
    # Mismatched pairs
    persons = list(os.listdir(lfw_dir))[:100]
    for i in range(0, len(persons)-1, 2):
        p1_dir = os.path.join(lfw_dir, persons[i])
        p2_dir = os.path.join(lfw_dir, persons[i+1])
        
        if os.path.isdir(p1_dir) and os.path.isdir(p2_dir):
            imgs1 = [os.path.join(p1_dir, f) 
                    for f in os.listdir(p1_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            imgs2 = [os.path.join(p2_dir, f) 
                    for f in os.listdir(p2_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if imgs1 and imgs2:
                test_pairs.append((imgs1[0], imgs2[0], False))
    
    # Collect fake images
    if os.path.exists(nuaa_fake):
        for root, dirs, files in os.walk(nuaa_fake):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fake_images.append(os.path.join(root, file))
                    if len(fake_images) >= 100:
                        break
    
    # Run studies
    report = ablation.generate_ablation_report(test_pairs, real_images, fake_images)
    ablation.save_ablation_report(report)
    
    return report


if __name__ == '__main__':
    run_ablation_studies()
