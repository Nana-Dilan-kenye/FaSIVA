"""
Comprehensive FaSIVA System Evaluation
Includes identification (CMC curves), verification (ROC/EER), 
super-resolution metrics, and cross-dataset liveness evaluation
"""
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from scipy.spatial.distance import cdist
from typing import Dict, Tuple, List
from tqdm import tqdm
import json
from datetime import datetime

from config import *
from main import FaSIVA
from feature_extraction import feature_extractor
from utils import load_image, euclidean_distance, get_resolution
from super_resolution import get_fsrcnn_model


class IdentificationEvaluator:
    """Evaluate identification performance with CMC curves and rank-1 recognition"""
    
    def __init__(self):
        self.fasiva = FaSIVA()
        self.gallery_features_f = []
        self.gallery_features_e = []
        self.gallery_identities = []
        self.probe_features_f = []
        self.probe_features_e = []
        self.probe_identities = []
        
    def build_gallery(self, gallery_dir: str, max_samples_per_id: int = 5):
        """Build gallery from dataset directory"""
        print(f"\n{'=' * 60}")
        print("BUILDING IDENTIFICATION GALLERY")
        print(f"{'=' * 60}")
        
        identity_count = 0
        image_count = 0
        
        for person_name in sorted(os.listdir(gallery_dir)):
            person_dir = os.path.join(gallery_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            identity_count += 1
            samples = 0
            
            for img_name in sorted(os.listdir(person_dir)):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_path = os.path.join(person_dir, img_name)
                try:
                    image = load_image(img_path)
                    faces = self.fasiva.face_detector.detect_faces(image)
                    
                    if faces:
                        face_crop = faces[0]['face']
                        f_vec, e_vec = feature_extractor.extract_signature_features(face_crop)
                        
                        self.gallery_features_f.append(f_vec)
                        self.gallery_features_e.append(e_vec)
                        self.gallery_identities.append(person_name)
                        image_count += 1
                        samples += 1
                        
                        if samples >= max_samples_per_id:
                            break
                except Exception as e:
                    continue
        
        self.gallery_features_f = np.array(self.gallery_features_f)
        self.gallery_features_e = np.array(self.gallery_features_e)
        
        print(f"✓ Gallery built: {image_count} images from {identity_count} identities")
        return image_count, identity_count
    
    def evaluate_on_pairs(self, pairs_file: str, test_dir: str) -> Dict:
        """Evaluate on matched/mismatched face pairs"""
        print(f"\n{'=' * 60}")
        print("IDENTIFICATION EVALUATION ON PAIRS")
        print(f"{'=' * 60}")
        
        results = {
            'genuine_distances': [],
            'impostor_distances': [],
            'genuine_matches': 0,
            'total_genuine': 0,
            'impostor_matches': 0,
            'total_impostor': 0
        }
        
        try:
            with open(pairs_file, 'r') as f:
                for line in tqdm(f, desc="Processing pairs"):
                    parts = line.strip().split()
                    
                    if len(parts) == 3:  # Matched pair
                        person, idx1, idx2 = parts
                        img1_path = os.path.join(test_dir, person, f"{person}_{int(idx1):04d}.jpg")
                        img2_path = os.path.join(test_dir, person, f"{person}_{int(idx2):04d}.jpg")
                        is_same = True
                    elif len(parts) == 4:  # Mismatched pair
                        person1, idx1, person2, idx2 = parts
                        img1_path = os.path.join(test_dir, person1, f"{person1}_{int(idx1):04d}.jpg")
                        img2_path = os.path.join(test_dir, person2, f"{person2}_{int(idx2):04d}.jpg")
                        is_same = False
                    else:
                        continue
                    
                    try:
                        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                            continue
                            
                        # Extract features
                        img1 = load_image(img1_path)
                        img2 = load_image(img2_path)
                        
                        faces1 = self.fasiva.face_detector.detect_faces(img1)
                        faces2 = self.fasiva.face_detector.detect_faces(img2)
                        
                        if faces1 and faces2:
                            f1_vec, _ = feature_extractor.extract_signature_features(faces1[0]['face'])
                            f2_vec, _ = feature_extractor.extract_signature_features(faces2[0]['face'])
                            
                            dist = euclidean_distance(f1_vec, f2_vec)
                            
                            if is_same:
                                results['genuine_distances'].append(dist)
                                results['total_genuine'] += 1
                                if dist <= IDENTIFICATION_THRESHOLD:
                                    results['genuine_matches'] += 1
                            else:
                                results['impostor_distances'].append(dist)
                                results['total_impostor'] += 1
                                if dist > IDENTIFICATION_THRESHOLD:
                                    results['impostor_matches'] += 1
                    except:
                        continue
        except FileNotFoundError:
            print(f"Pairs file not found: {pairs_file}")
            return results
        
        # Calculate metrics
        if results['total_genuine'] > 0:
            results['genuine_accuracy'] = results['genuine_matches'] / results['total_genuine']
        if results['total_impostor'] > 0:
            results['impostor_accuracy'] = results['impostor_matches'] / results['total_impostor']
        
        print(f"✓ Genuine pairs: {results['genuine_matches']}/{results['total_genuine']} correct")
        print(f"✓ Impostor pairs: {results['impostor_matches']}/{results['total_impostor']} rejected")
        
        return results
    
    def compute_cmc_curve(self, test_dir: str) -> Tuple[List[float], List[int]]:
        """Compute Cumulative Matching Characteristic (CMC) curve"""
        print(f"\n{'=' * 60}")
        print("COMPUTING CMC CURVE")
        print(f"{'=' * 60}")
        
        ranks_correct = []
        test_identities = []
        test_features_f = []
        
        # Load test samples
        for person_name in sorted(os.listdir(test_dir)):
            person_dir = os.path.join(test_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            for img_name in sorted(os.listdir(person_dir))[:2]:  # Use max 2 samples per person
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_path = os.path.join(person_dir, img_name)
                try:
                    image = load_image(img_path)
                    faces = self.fasiva.face_detector.detect_faces(image)
                    
                    if faces:
                        face_crop = faces[0]['face']
                        f_vec, _ = feature_extractor.extract_signature_features(face_crop)
                        test_features_f.append(f_vec)
                        test_identities.append(person_name)
                except:
                    continue
        
        if not test_features_f or len(self.gallery_features_f) == 0:
            print("Insufficient data for CMC computation")
            return [], []
        
        test_features_f = np.array(test_features_f)
        
        # Compute distances
        distances = cdist(test_features_f, self.gallery_features_f, metric='euclidean')
        
        # For each test sample, find rank of correct identity
        max_rank = min(100, len(np.unique(self.gallery_identities)))
        cmc_values = np.zeros(max_rank)
        
        for i, test_identity in enumerate(test_identities):
            # Get distances for this test sample
            dist_row = distances[i]
            
            # Sort by distance (ascending)
            sorted_indices = np.argsort(dist_row)
            
            # Find rank of first correct match
            for rank, idx in enumerate(sorted_indices):
                if self.gallery_identities[idx] == test_identity:
                    if rank < max_rank:
                        cmc_values[rank:] += 1
                    break
            ranks_correct.append(rank + 1 if rank < max_rank else max_rank + 1)
        
        # Normalize CMC curve
        num_tests = len(test_identities)
        cmc_curve = cmc_values / num_tests if num_tests > 0 else cmc_values
        
        # Compute statistics
        rank_1_accuracy = cmc_curve[0] if len(cmc_curve) > 0 else 0
        rank_5_accuracy = cmc_curve[4] if len(cmc_curve) > 4 else 1.0
        
        print(f"✓ Rank-1 Recognition Rate: {rank_1_accuracy:.4f}")
        print(f"✓ Rank-5 Recognition Rate: {rank_5_accuracy:.4f}")
        print(f"✓ Average Rank: {np.mean(ranks_correct):.2f}")
        print(f"✓ Median Rank: {np.median(ranks_correct):.2f}")
        
        return cmc_curve.tolist(), ranks_correct
    
    def plot_cmc_curve(self, cmc_curve: List[float], output_path: str = "cmc_curve.png"):
        """Plot CMC curve"""
        if not cmc_curve:
            return
        
        plt.figure(figsize=(10, 6))
        ranks = list(range(1, len(cmc_curve) + 1))
        plt.plot(ranks, cmc_curve, 'b-', linewidth=2, marker='o')
        plt.xlabel('Rank', fontsize=12)
        plt.ylabel('Recognition Rate', fontsize=12)
        plt.title('Cumulative Matching Characteristic (CMC) Curve', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim(1, min(20, len(cmc_curve)))
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"✓ CMC curve saved to {output_path}")
        plt.close()


class VerificationEvaluator:
    """Evaluate verification performance with ROC curves and EER"""
    
    def __init__(self):
        self.fasiva = FaSIVA()
    
    def evaluate_verification(self, pairs_file: str, test_dir: str) -> Dict:
        """Evaluate verification with ROC analysis"""
        print(f"\n{'=' * 60}")
        print("VERIFICATION EVALUATION (ROC/EER)")
        print(f"{'=' * 60}")
        
        genuine_scores = []
        impostor_scores = []
        
        try:
            with open(pairs_file, 'r') as f:
                for line in tqdm(f, desc="Computing verification scores"):
                    parts = line.strip().split()
                    
                    if len(parts) == 3:
                        person, idx1, idx2 = parts
                        img1_path = os.path.join(test_dir, person, f"{person}_{int(idx1):04d}.jpg")
                        img2_path = os.path.join(test_dir, person, f"{person}_{int(idx2):04d}.jpg")
                        is_same = True
                    elif len(parts) == 4:
                        person1, idx1, person2, idx2 = parts
                        img1_path = os.path.join(test_dir, person1, f"{person1}_{int(idx1):04d}.jpg")
                        img2_path = os.path.join(test_dir, person2, f"{person2}_{int(idx2):04d}.jpg")
                        is_same = False
                    else:
                        continue
                    
                    try:
                        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                            continue
                        
                        img1 = load_image(img1_path)
                        img2 = load_image(img2_path)
                        
                        faces1 = self.fasiva.face_detector.detect_faces(img1)
                        faces2 = self.fasiva.face_detector.detect_faces(img2)
                        
                        if faces1 and faces2:
                            _, e1_vec = feature_extractor.extract_signature_features(faces1[0]['face'])
                            _, e2_vec = feature_extractor.extract_signature_features(faces2[0]['face'])
                            
                            # Use cosine similarity as score (higher = more similar)
                            dist = euclidean_distance(e1_vec, e2_vec)
                            score = 1.0 / (1.0 + dist)  # Convert distance to similarity
                            
                            if is_same:
                                genuine_scores.append(score)
                            else:
                                impostor_scores.append(score)
                    except:
                        continue
        except FileNotFoundError:
            print(f"Pairs file not found: {pairs_file}")
            return {}
        
        if not genuine_scores or not impostor_scores:
            print("Insufficient data for ROC analysis")
            return {}
        
        # Compute ROC curve
        labels = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
        scores = np.array(genuine_scores + impostor_scores)
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # Compute EER
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer = fpr[eer_idx]
        eer_threshold = thresholds[eer_idx]
        
        # Compute GAR at specific FAR levels
        far_levels = [0.001, 0.01, 0.1]
        gar_at_far = {}
        
        for far_level in far_levels:
            far_idx = np.argmin(np.abs(fpr - far_level))
            if far_idx < len(tpr):
                gar_at_far[f"FAR={far_level}"] = tpr[far_idx]
        
        results = {
            'genuine_scores': genuine_scores,
            'impostor_scores': impostor_scores,
            'roc_auc': float(roc_auc),
            'eer': float(eer),
            'eer_threshold': float(eer_threshold),
            'gar_at_far': {k: float(v) for k, v in gar_at_far.items()},
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        print(f"✓ ROC-AUC Score: {roc_auc:.4f}")
        print(f"✓ Equal Error Rate (EER): {eer:.4f}")
        print(f"✓ EER Threshold: {eer_threshold:.4f}")
        for k, v in results['gar_at_far'].items():
            print(f"✓ GAR at {k}: {v:.4f}")
        
        return results
    
    def plot_roc_curve(self, fpr: List[float], tpr: List[float], roc_auc: float, 
                       output_path: str = "roc_curve.png"):
        """Plot ROC curve"""
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve for Verification', fontsize=14)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"✓ ROC curve saved to {output_path}")
        plt.close()


class SuperResolutionEvaluator:
    """Evaluate super-resolution quality using PSNR and SSIM metrics"""
    
    def __init__(self):
        try:
            self.fsrcnn_model = get_fsrcnn_model()
            self.device = DEVICE
        except:
            print("FSRCNN model not available")
            self.fsrcnn_model = None
    
    def calculate_psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def calculate_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Structural Similarity Index Map (SSIM)"""
        # Ensure same size
        if original.shape != reconstructed.shape:
            reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]))
        
        c1, c2 = 6.5025, 58.5225
        
        orig_float = original.astype(float)
        recon_float = reconstructed.astype(float)
        
        mean1 = cv2.blur(orig_float, (11, 11))
        mean2 = cv2.blur(recon_float, (11, 11))
        
        mean1_sq = mean1 ** 2
        mean2_sq = mean2 ** 2
        mean1_mean2 = mean1 * mean2
        
        sigma1_sq = cv2.blur(orig_float ** 2, (11, 11)) - mean1_sq
        sigma2_sq = cv2.blur(recon_float ** 2, (11, 11)) - mean2_sq
        sigma12 = cv2.blur(orig_float * recon_float, (11, 11)) - mean1_mean2
        
        numerator = (2 * mean1_mean2 + c1) * (2 * sigma12 + c2)
        denominator = (mean1_sq + mean2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        
        ssim_map = numerator / denominator
        ssim = np.mean(ssim_map)
        return ssim
    
    def evaluate_on_dataset(self, test_dir: str, scale_factor: int = 4) -> Dict:
        """Evaluate super-resolution on test images"""
        print(f"\n{'=' * 60}")
        print("SUPER-RESOLUTION QUALITY EVALUATION")
        print(f"{'=' * 60}")
        
        if not self.fsrcnn_model:
            print("FSRCNN model not available")
            return {}
        
        results = {
            'psnr_scores': [],
            'ssim_scores': [],
            'images_processed': 0
        }
        
        # Collect test images
        test_images = []
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(root, file))
        
        for img_path in tqdm(test_images[:100], desc="Evaluating SR quality"):
            try:
                # Load high-res image
                high_res = cv2.imread(img_path)
                if high_res is None:
                    continue
                
                # Create low-res by downsampling
                low_res = cv2.resize(high_res, None, fx=1/scale_factor, fy=1/scale_factor)
                
                # Apply super-resolution
                with torch.no_grad():
                    # Convert to tensor
                    low_res_tensor = torch.from_numpy(low_res.astype(np.float32) / 255.0)
                    low_res_tensor = low_res_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    # Super-resolve
                    sr_tensor = self.fsrcnn_model(low_res_tensor)
                    sr_image = (sr_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                # Calculate metrics on Y channel (luminance)
                high_res_y = cv2.cvtColor(high_res, cv2.COLOR_BGR2GRAY)
                sr_y = cv2.cvtColor(sr_image, cv2.COLOR_BGR2GRAY)
                
                psnr = self.calculate_psnr(high_res_y, sr_y)
                ssim = self.calculate_ssim(high_res_y, sr_y)
                
                results['psnr_scores'].append(psnr)
                results['ssim_scores'].append(ssim)
                results['images_processed'] += 1
            except Exception as e:
                continue
        
        if results['psnr_scores']:
            results['avg_psnr'] = float(np.mean(results['psnr_scores']))
            results['std_psnr'] = float(np.std(results['psnr_scores']))
            results['avg_ssim'] = float(np.mean(results['ssim_scores']))
            results['std_ssim'] = float(np.std(results['ssim_scores']))
            
            print(f"✓ Images processed: {results['images_processed']}")
            print(f"✓ Average PSNR: {results['avg_psnr']:.2f} dB")
            print(f"✓ PSNR Std Dev: {results['std_psnr']:.2f} dB")
            print(f"✓ Average SSIM: {results['avg_ssim']:.4f}")
            print(f"✓ SSIM Std Dev: {results['std_ssim']:.4f}")
        
        return results


class CrossDatasetLivenessEvaluator:
    """Evaluate liveness detection across multiple datasets"""
    
    def __init__(self):
        self.fasiva = FaSIVA()
    
    def evaluate_dataset(self, dataset_name: str, real_dir: str, fake_dir: str) -> Dict:
        """Evaluate liveness on single dataset"""
        print(f"\n  Evaluating {dataset_name}...")
        
        results = {
            'dataset': dataset_name,
            'real_correct': 0,
            'real_total': 0,
            'fake_correct': 0,
            'fake_total': 0,
            'confidence_scores': []
        }
        
        # Evaluate real faces
        real_images = []
        if os.path.exists(real_dir):
            for root, dirs, files in os.walk(real_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        real_images.append(os.path.join(root, file))
        
        for img_path in tqdm(real_images[:200], desc=f"Real faces ({dataset_name})", leave=False):
            try:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                is_live, confidence = self.fasiva.liveness_detector.detect_liveness(image)
                results['real_total'] += 1
                if is_live:
                    results['real_correct'] += 1
                results['confidence_scores'].append(('real', confidence))
            except:
                continue
        
        # Evaluate fake faces
        fake_images = []
        if os.path.exists(fake_dir):
            for root, dirs, files in os.walk(fake_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        fake_images.append(os.path.join(root, file))
        
        for img_path in tqdm(fake_images[:200], desc=f"Fake faces ({dataset_name})", leave=False):
            try:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                is_live, confidence = self.fasiva.liveness_detector.detect_liveness(image)
                results['fake_total'] += 1
                if not is_live:
                    results['fake_correct'] += 1
                results['confidence_scores'].append(('fake', confidence))
            except:
                continue
        
        # Calculate metrics
        if results['real_total'] > 0:
            results['real_accuracy'] = results['real_correct'] / results['real_total']
        if results['fake_total'] > 0:
            results['fake_accuracy'] = results['fake_correct'] / results['fake_total']
        
        if results['real_total'] > 0 and results['fake_total'] > 0:
            results['average_accuracy'] = (results['real_accuracy'] + results['fake_accuracy']) / 2
        
        return results
    
    def evaluate_cross_dataset(self) -> Dict:
        """Evaluate across all available liveness datasets"""
        print(f"\n{'=' * 60}")
        print("CROSS-DATASET LIVENESS EVALUATION")
        print(f"{'=' * 60}")
        
        all_results = {}
        
        # NUAA
        nuaa_real = os.path.join(DATASETS_DIR, 'CASIA2', 'ClientRaw')
        nuaa_fake = os.path.join(DATASETS_DIR, 'CASIA2', 'ImposterRaw')
        if os.path.exists(nuaa_real) and os.path.exists(nuaa_fake):
            all_results['NUAA'] = self.evaluate_dataset('NUAA', nuaa_real, nuaa_fake)
        
        # Replay-Attack
        replay_real = os.path.join(DATASETS_DIR, 'replay_attack_dataset', 'real')
        replay_fake = os.path.join(DATASETS_DIR, 'replay_attack_dataset', 'fake')
        if os.path.exists(replay_real) and os.path.exists(replay_fake):
            all_results['Replay-Attack'] = self.evaluate_dataset('Replay-Attack', replay_real, replay_fake)
        
        # CASIA-2
        casia_real = os.path.join(DATASETS_DIR, 'CASIA2', 'real')
        casia_fake = os.path.join(DATASETS_DIR, 'CASIA2', 'fake')
        if os.path.exists(casia_real) and os.path.exists(casia_fake):
            all_results['CASIA-2'] = self.evaluate_dataset('CASIA-2', casia_real, casia_fake)
        
        # Compute cross-dataset statistics
        print(f"\n  Summary:")
        for dataset, results in all_results.items():
            print(f"    {dataset}:")
            print(f"      Real accuracy: {results.get('real_accuracy', 0):.4f}")
            print(f"      Fake accuracy: {results.get('fake_accuracy', 0):.4f}")
            print(f"      Overall accuracy: {results.get('average_accuracy', 0):.4f}")
        
        return all_results


class AuthenticationScenarioTester:
    """Test end-to-end authentication scenarios"""
    
    def __init__(self):
        self.fasiva = FaSIVA()
    
    def test_complete_authentication(self, enrolled_image: str, test_images: List[str]) -> Dict:
        """Test complete authentication pipeline"""
        print(f"\n{'=' * 60}")
        print("END-TO-END AUTHENTICATION SCENARIO TEST")
        print(f"{'=' * 60}")
        
        results = {
            'total_tests': len(test_images),
            'successful_authentications': 0,
            'failed_authentications': 0,
            'spoofing_attempts_detected': 0,
            'false_rejections': 0,
            'scenarios': []
        }
        
        # Enroll the identity
        print(f"\nEnrolling identity from {os.path.basename(enrolled_image)}...")
        
        try:
            enrolled_img = load_image(enrolled_image)
            enrolled_faces = self.fasiva.face_detector.detect_faces(enrolled_img)
            
            if not enrolled_faces:
                print("Failed to detect face in enrolled image")
                return results
            
            enrolled_data = self.fasiva.process_image(enrolled_image, save_signature=True)
        except Exception as e:
            print(f"Enrollment failed: {e}")
            return results
        
        # Test authentication scenarios
        for test_img_path in test_images:
            scenario = {
                'image': os.path.basename(test_img_path),
                'authenticated': False,
                'reason': ''
            }
            
            try:
                auth_result = self.fasiva.authenticate_person(test_img_path)
                
                if auth_result.get('authentication', {}).get('success', False):
                    scenario['authenticated'] = True
                    scenario['reason'] = 'Successful authentication'
                    results['successful_authentications'] += 1
                else:
                    auth_data = auth_result.get('authentication', {})
                    stage_failed = auth_data.get('stage_failed', '')
                    
                    if stage_failed == 'liveness':
                        scenario['reason'] = 'Spoofing attack detected'
                        results['spoofing_attempts_detected'] += 1
                    else:
                        scenario['reason'] = f'Authentication failed at {stage_failed}'
                        results['false_rejections'] += 1
                    results['failed_authentications'] += 1
            except Exception as e:
                scenario['reason'] = f'Error: {str(e)}'
                results['failed_authentications'] += 1
            
            results['scenarios'].append(scenario)
        
        # Summary
        print(f"\n  Results Summary:")
        print(f"    Total tests: {results['total_tests']}")
        print(f"    Successful: {results['successful_authentications']}")
        print(f"    Failed: {results['failed_authentications']}")
        print(f"    Spoofing detected: {results['spoofing_attempts_detected']}")
        
        return results


def run_comprehensive_evaluation():
    """Run all evaluation components"""
    
    print("\n" + "=" * 70)
    print("FASIVA COMPREHENSIVE SYSTEM EVALUATION")
    print("=" * 70)
    
    evaluation_report = {
        'timestamp': datetime.now().isoformat(),
        'identification': {},
        'verification': {},
        'super_resolution': {},
        'cross_dataset_liveness': {},
        'authentication': {}
    }
    
    # 1. Identification Evaluation
    try:
        id_eval = IdentificationEvaluator()
        lfw_dir = os.path.join(DATASETS_DIR, 'lfw-deepfunneled', 'lfw-deepfunneled')
        
        if os.path.exists(lfw_dir):
            id_eval.build_gallery(lfw_dir, max_samples_per_id=3)
            
            # CMC curve
            cmc_curve, ranks = id_eval.compute_cmc_curve(lfw_dir)
            id_eval.plot_cmc_curve(cmc_curve)
            
            evaluation_report['identification']['cmc_curve'] = cmc_curve[:20]
            evaluation_report['identification']['rank_1'] = cmc_curve[0] if cmc_curve else 0
            evaluation_report['identification']['status'] = 'completed'
    except Exception as e:
        print(f"Identification evaluation error: {e}")
        evaluation_report['identification']['error'] = str(e)
    
    # 2. Verification Evaluation
    try:
        ver_eval = VerificationEvaluator()
        lfw_pairs = os.path.join(DATASETS_DIR, 'pairs.csv')
        lfw_test_dir = os.path.join(DATASETS_DIR, 'lfw-deepfunneled', 'lfw-deepfunneled')
        
        if os.path.exists(lfw_pairs) and os.path.exists(lfw_test_dir):
            ver_results = ver_eval.evaluate_verification(lfw_pairs, lfw_test_dir)
            
            if 'fpr' in ver_results:
                ver_eval.plot_roc_curve(ver_results['fpr'], ver_results['tpr'], ver_results['roc_auc'])
            
            evaluation_report['verification'].update(ver_results)
            evaluation_report['verification']['status'] = 'completed'
    except Exception as e:
        print(f"Verification evaluation error: {e}")
        evaluation_report['verification']['error'] = str(e)
    
    # 3. Super-Resolution Evaluation
    try:
        sr_eval = SuperResolutionEvaluator()
        lfw_dir = os.path.join(DATASETS_DIR, 'lfw-deepfunneled', 'lfw-deepfunneled')
        
        if os.path.exists(lfw_dir):
            sr_results = sr_eval.evaluate_on_dataset(lfw_dir)
            evaluation_report['super_resolution'].update(sr_results)
            evaluation_report['super_resolution']['status'] = 'completed'
    except Exception as e:
        print(f"Super-resolution evaluation error: {e}")
        evaluation_report['super_resolution']['error'] = str(e)
    
    # 4. Cross-Dataset Liveness Evaluation
    try:
        liveness_eval = CrossDatasetLivenessEvaluator()
        cross_results = liveness_eval.evaluate_cross_dataset()
        
        for dataset, results in cross_results.items():
            evaluation_report['cross_dataset_liveness'][dataset] = {
                'real_accuracy': results.get('real_accuracy', 0),
                'fake_accuracy': results.get('fake_accuracy', 0),
                'average_accuracy': results.get('average_accuracy', 0)
            }
        
        evaluation_report['cross_dataset_liveness']['status'] = 'completed'
    except Exception as e:
        print(f"Cross-dataset liveness evaluation error: {e}")
        evaluation_report['cross_dataset_liveness']['error'] = str(e)
    
    # 5. Authentication Scenario Testing
    try:
        auth_tester = AuthenticationScenarioTester()
        test_dir = os.path.join(DATASETS_DIR, 'test')
        
        if os.path.exists(test_dir):
            test_files = []
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_files.append(os.path.join(root, file))
            
            if test_files:
                auth_results = auth_tester.test_complete_authentication(
                    test_files[0],
                    test_files[1:6]
                )
                evaluation_report['authentication'].update({
                    'total_tests': auth_results['total_tests'],
                    'successful': auth_results['successful_authentications'],
                    'failed': auth_results['failed_authentications'],
                    'status': 'completed'
                })
    except Exception as e:
        print(f"Authentication evaluation error: {e}")
        evaluation_report['authentication']['error'] = str(e)
    
    # Save comprehensive report
    report_path = os.path.join(os.getcwd(), 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"✓ Comprehensive evaluation completed")
    print(f"✓ Report saved to: {report_path}")
    print(f"{'=' * 70}\n")
    
    return evaluation_report


if __name__ == '__main__':
    run_comprehensive_evaluation()
