"""
Enhanced Verification Module for FaSIVA
Includes threshold optimization, detailed ROC analysis, and improved metrics
"""
import numpy as np
from typing import Tuple, Dict, List
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_curve, auc
import json
import os

from config import *
from utils import euclidean_distance


class VerificationOptimizer:
    """Optimize verification thresholds and compute detailed metrics"""
    
    def __init__(self):
        self.optimal_threshold = VERIFICATION_THRESHOLD
        self.eer_threshold = None
        self.metrics_history = []
    
    def compute_confusion_matrix(self, genuine_scores: np.ndarray, 
                                impostor_scores: np.ndarray,
                                threshold: float) -> Dict:
        """Compute confusion matrix at given threshold"""
        # For FaceNet embeddings, use distance-based decision
        genuine_correct = np.sum(genuine_scores <= threshold)
        genuine_total = len(genuine_scores)
        genuines_rejected = genuine_total - genuine_correct
        
        impostor_correct = np.sum(impostor_scores > threshold)
        impostor_total = len(impostor_scores)
        impostors_accepted = impostor_total - impostor_correct
        
        return {
            'TP': genuine_correct,  # Genuine accepted
            'FN': genuines_rejected,  # Genuine rejected
            'FP': impostors_accepted,  # Impostor accepted
            'TN': impostor_correct,  # Impostor rejected
            'FAR': impostors_accepted / impostor_total if impostor_total > 0 else 0,
            'FRR': genuines_rejected / genuine_total if genuine_total > 0 else 0,
            'GAR': genuine_correct / genuine_total if genuine_total > 0 else 0,  # Genuine Acceptance Rate
            'TAR': genuine_correct / genuine_total if genuine_total > 0 else 0   # True Acceptance Rate
        }
    
    def find_optimal_threshold(self, genuine_scores: np.ndarray,
                               impostor_scores: np.ndarray) -> Tuple[float, float]:
        """Find threshold that minimizes FAR + FRR"""
        
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        thresholds = np.sort(np.unique(all_scores))
        
        best_threshold = VERIFICATION_THRESHOLD
        best_metric = float('inf')
        
        metrics_at_thresholds = []
        
        for threshold in thresholds:
            cm = self.compute_confusion_matrix(genuine_scores, impostor_scores, threshold)
            metric = cm['FAR'] + cm['FRR']
            
            metrics_at_thresholds.append({
                'threshold': threshold,
                'FAR': cm['FAR'],
                'FRR': cm['FRR'],
                'GAR': cm['GAR'],
                'FAR_FRR_sum': metric
            })
            
            if metric < best_metric:
                best_metric = metric
                best_threshold = threshold
        
        self.metrics_history = metrics_at_thresholds
        return best_threshold, best_metric
    
    def find_eer_threshold(self, genuine_scores: np.ndarray,
                          impostor_scores: np.ndarray) -> Tuple[float, float]:
        """Find threshold where FAR = FRR (Equal Error Rate)"""
        
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        thresholds = np.sort(np.unique(all_scores))
        
        best_threshold = VERIFICATION_THRESHOLD
        min_diff = float('inf')
        eer_value = 0
        
        for threshold in thresholds:
            cm = self.compute_confusion_matrix(genuine_scores, impostor_scores, threshold)
            diff = abs(cm['FAR'] - cm['FRR'])
            
            if diff < min_diff:
                min_diff = diff
                best_threshold = threshold
                eer_value = (cm['FAR'] + cm['FRR']) / 2
        
        self.eer_threshold = best_threshold
        return best_threshold, eer_value
    
    def compute_detection_error_tradeoff(self, genuine_scores: np.ndarray,
                                         impostor_scores: np.ndarray) -> Dict:
        """Compute DET (Detection Error Tradeoff) metrics"""
        
        fpr, fnr, thresholds = self._compute_error_tradeoff(genuine_scores, impostor_scores)
        
        return {
            'fpr': fpr.tolist(),
            'fnr': fnr.tolist(),
            'thresholds': thresholds.tolist()
        }
    
    def _compute_error_tradeoff(self, genuine_scores: np.ndarray,
                               impostor_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Internal method to compute error tradeoff"""
        
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        thresholds = np.sort(np.unique(all_scores))
        
        fpr_list = []
        fnr_list = []
        
        for threshold in thresholds:
            cm = self.compute_confusion_matrix(genuine_scores, impostor_scores, threshold)
            fpr_list.append(cm['FAR'])
            fnr_list.append(cm['FRR'])
        
        return np.array(fpr_list), np.array(fnr_list), thresholds
    
    def generate_verification_report(self, genuine_scores: np.ndarray,
                                    impostor_scores: np.ndarray) -> Dict:
        """Generate comprehensive verification report"""
        
        # Find optimal threshold
        opt_threshold, opt_metric = self.find_optimal_threshold(genuine_scores, impostor_scores)
        
        # Find EER threshold
        eer_threshold, eer = self.find_eer_threshold(genuine_scores, impostor_scores)
        
        # Compute metrics at default threshold
        default_cm = self.compute_confusion_matrix(genuine_scores, impostor_scores, 
                                                   VERIFICATION_THRESHOLD)
        
        # Compute metrics at optimal threshold
        optimal_cm = self.compute_confusion_matrix(genuine_scores, impostor_scores, 
                                                   opt_threshold)
        
        # Compute metrics at EER threshold
        eer_cm = self.compute_confusion_matrix(genuine_scores, impostor_scores, 
                                              eer_threshold)
        
        # Compute ROC-AUC
        labels = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
        scores = np.array(list(genuine_scores) + list(impostor_scores))
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # GAR at specific FAR levels
        gar_table = self._compute_gar_at_far(genuine_scores, impostor_scores)
        
        report = {
            'summary': {
                'total_genuine': len(genuine_scores),
                'total_impostor': len(impostor_scores),
                'roc_auc': float(roc_auc)
            },
            'default_threshold': {
                'threshold': VERIFICATION_THRESHOLD,
                'FAR': float(default_cm['FAR']),
                'FRR': float(default_cm['FRR']),
                'GAR': float(default_cm['GAR']),
                'TAR': float(default_cm['TAR'])
            },
            'optimal_threshold': {
                'threshold': float(opt_threshold),
                'FAR': float(optimal_cm['FAR']),
                'FRR': float(optimal_cm['FRR']),
                'GAR': float(optimal_cm['GAR']),
                'TAR': float(optimal_cm['TAR']),
                'FAR_FRR_sum': float(opt_metric)
            },
            'eer_threshold': {
                'threshold': float(eer_threshold),
                'EER': float(eer),
                'FAR': float(eer_cm['FAR']),
                'FRR': float(eer_cm['FRR']),
                'GAR': float(eer_cm['GAR'])
            },
            'gar_at_far_table': gar_table
        }
        
        return report
    
    def _compute_gar_at_far(self, genuine_scores: np.ndarray,
                           impostor_scores: np.ndarray) -> Dict:
        """Compute GAR at specific FAR operating points"""
        
        far_targets = [0.001, 0.01, 0.05, 0.1]
        gar_table = {}
        
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        thresholds = np.sort(np.unique(all_scores))
        
        for far_target in far_targets:
            best_threshold = None
            best_gar = 0
            
            for threshold in thresholds:
                cm = self.compute_confusion_matrix(genuine_scores, impostor_scores, threshold)
                
                if abs(cm['FAR'] - far_target) < 0.01:  # Close to target FAR
                    if cm['GAR'] > best_gar:
                        best_gar = cm['GAR']
                        best_threshold = threshold
            
            if best_threshold is not None:
                gar_table[f'GAR@FAR={far_target}'] = float(best_gar)
        
        return gar_table


class VerificationModule:
    """Main verification module with complete pipeline"""
    
    def __init__(self):
        self.optimizer = VerificationOptimizer()
        self.threshold = VERIFICATION_THRESHOLD
    
    def verify_identity(self, enrolled_embedding: np.ndarray,
                       test_embedding: np.ndarray,
                       threshold: float = None) -> Tuple[bool, float]:
        """
        Verify if test embedding matches enrolled embedding
        
        Args:
            enrolled_embedding: Feature vector from enrollment (E vector)
            test_embedding: Feature vector from test (E vector)
            threshold: Distance threshold (uses default if None)
        
        Returns:
            (verified: bool, confidence: float)
        """
        
        if threshold is None:
            threshold = self.threshold
        
        # Compute distance
        distance = euclidean_distance(enrolled_embedding, test_embedding)
        
        # Similarity score (inverse of distance)
        confidence = 1.0 / (1.0 + distance)
        
        verified = distance <= threshold
        
        return verified, confidence
    
    def verify_batch(self, enrolled_embedding: np.ndarray,
                    test_embeddings: List[np.ndarray],
                    threshold: float = None) -> List[Tuple[bool, float]]:
        """Verify multiple test embeddings against one enrolled embedding"""
        
        results = []
        for test_embedding in test_embeddings:
            verified, confidence = self.verify_identity(enrolled_embedding, test_embedding, threshold)
            results.append((verified, confidence))
        
        return results
    
    def optimize_threshold(self, genuine_distances: np.ndarray,
                          impostor_distances: np.ndarray) -> Dict:
        """
        Optimize verification threshold based on score distributions
        
        Args:
            genuine_distances: Array of distances for genuine pairs
            impostor_distances: Array of distances for impostor pairs
        
        Returns:
            Optimization results dictionary
        """
        
        report = self.optimizer.generate_verification_report(genuine_distances, impostor_distances)
        self.threshold = report['optimal_threshold']['threshold']
        
        return report
    
    def set_threshold(self, threshold: float):
        """Set custom verification threshold"""
        self.threshold = threshold
    
    def save_report(self, report: Dict, output_path: str):
        """Save verification report to JSON"""
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Verification report saved to {output_path}")


class AuthenticationVector:
    """
    FaSIVA Authentication Vector A = [a1, a2]
    a1: Reflection-based liveness confidence
    a2: Eye blink detection confidence
    """
    
    def __init__(self):
        self.a1 = 0.0  # Reflection confidence
        self.a2 = 0.0  # Eye blink confidence
    
    def compute_authentication_score(self) -> float:
        """
        Compute overall authentication score
        Score = α * a1 + (1-α) * a2
        where α ∈ [0, 1] is weighting factor
        """
        alpha = 0.5  # Equal weight (configurable)
        return alpha * self.a1 + (1 - alpha) * self.a2
    
    def set_reflection_confidence(self, confidence: float):
        """Set reflection-based liveness confidence"""
        self.a1 = np.clip(confidence, 0.0, 1.0)
    
    def set_eye_blink_confidence(self, confidence: float):
        """Set eye blink detection confidence"""
        self.a2 = np.clip(confidence, 0.0, 1.0)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'a1_reflection': float(self.a1),
            'a2_eye_blink': float(self.a2),
            'authentication_score': self.compute_authentication_score()
        }


def print_verification_report(report: Dict):
    """Pretty print verification report"""
    
    print("\n" + "=" * 70)
    print("VERIFICATION PERFORMANCE REPORT")
    print("=" * 70)
    
    print(f"\nSummary:")
    print(f"  Total genuine pairs: {report['summary']['total_genuine']}")
    print(f"  Total impostor pairs: {report['summary']['total_impostor']}")
    print(f"  ROC-AUC Score: {report['summary']['roc_auc']:.4f}")
    
    print(f"\nAt Default Threshold ({report['default_threshold']['threshold']:.4f}):")
    print(f"  FAR: {report['default_threshold']['FAR']:.6f}")
    print(f"  FRR: {report['default_threshold']['FRR']:.6f}")
    print(f"  GAR: {report['default_threshold']['GAR']:.4f}")
    print(f"  TAR: {report['default_threshold']['TAR']:.4f}")
    
    print(f"\nAt Optimal Threshold ({report['optimal_threshold']['threshold']:.4f}):")
    print(f"  FAR: {report['optimal_threshold']['FAR']:.6f}")
    print(f"  FRR: {report['optimal_threshold']['FRR']:.6f}")
    print(f"  GAR: {report['optimal_threshold']['GAR']:.4f}")
    print(f"  TAR: {report['optimal_threshold']['TAR']:.4f}")
    print(f"  FAR+FRR: {report['optimal_threshold']['FAR_FRR_sum']:.6f}")
    
    print(f"\nAt EER Threshold ({report['eer_threshold']['threshold']:.4f}):")
    print(f"  EER: {report['eer_threshold']['EER']:.6f}")
    print(f"  FAR: {report['eer_threshold']['FAR']:.6f}")
    print(f"  FRR: {report['eer_threshold']['FRR']:.6f}")
    print(f"  GAR: {report['eer_threshold']['GAR']:.4f}")
    
    if report['gar_at_far_table']:
        print(f"\nGAR at Specific FAR Operating Points:")
        for key, value in report['gar_at_far_table'].items():
            print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 70 + "\n")
