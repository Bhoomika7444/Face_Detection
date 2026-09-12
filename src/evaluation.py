import os
import time
import numpy as np
from typing import Tuple, Dict, List
from .utils import load_image
from .detector import FaceDetector
from .embedder import FaceEmbedder
from .recognizer import FaceRecognizer

class FaceEvaluator:
    def __init__(self, detector: FaceDetector, embedder: FaceEmbedder, recognizer: FaceRecognizer):
        self.detector = detector
        self.embedder = embedder
        self.recognizer = recognizer
        
    def evaluate_directory(self, test_dir: str, threshold: float = None) -> Dict:
        """
        Evaluates the system on a given dataset directory.
        Expected structure:
        test_dir/
            known/
                PersonA/
                    img1.jpg
                PersonB/
                    img2.jpg
            unknown/
                StrangerA/
                    img3.jpg
        """
        if threshold is None:
            threshold = self.recognizer.default_threshold
            
        y_true = []
        y_pred = []
        y_scores = []
        
        start_time = time.time()
        processed_count = 0
        
        # Test Known
        known_dir = os.path.join(test_dir, 'known')
        if os.path.exists(known_dir):
            for identity in os.listdir(known_dir):
                person_dir = os.path.join(known_dir, identity)
                if not os.path.isdir(person_dir): continue
                
                for img_name in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_name)
                    pred_id, score = self._process_image(img_path, threshold)
                    if pred_id is not None:
                        y_true.append(identity)
                        y_pred.append(pred_id)
                        y_scores.append(score)
                        processed_count += 1
                        
        # Test Unknown
        unknown_dir = os.path.join(test_dir, 'unknown')
        if os.path.exists(unknown_dir):
            for identity in os.listdir(unknown_dir):
                person_dir = os.path.join(unknown_dir, identity)
                if not os.path.isdir(person_dir): continue
                
                for img_name in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_name)
                    pred_id, score = self._process_image(img_path, threshold)
                    if pred_id is not None:
                        y_true.append('UNKNOWN')
                        y_pred.append(pred_id)
                        y_scores.append(score)
                        processed_count += 1
                        
        end_time = time.time()
        
        metrics = self._calculate_metrics(y_true, y_pred)
        metrics['processed_count'] = processed_count
        metrics['time_taken'] = end_time - start_time
        metrics['threshold'] = threshold
        
        return metrics
        
    def _process_image(self, img_path: str, threshold: float) -> Tuple[str, float]:
        """Processes a single image and returns predicted id and score."""
        try:
            image = load_image(img_path)
            faces = self.detector.detect_faces(image)
            
            if not faces:
                return None, 0.0 # No face detected, skip
                
            # For evaluation, we assume 1 face per image. Take the first.
            face = faces[0]
            embedding = self.embedder.get_embedding(face['face_tensor'])
            
            result = self.recognizer.identify(embedding, threshold=threshold)
            return result['name'], result['similarity']
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None, 0.0

    def _calculate_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict:
        """Calculates standard metrics."""
        if not y_true:
            return {'error': 'No valid test data found'}
            
        correct_known = 0
        total_known = 0
        correct_unknown = 0
        total_unknown = 0
        false_accepts = 0 # Unknown person classified as known
        false_rejects = 0 # Known person classified as UNKNOWN (or wrong person)
        
        for yt, yp in zip(y_true, y_pred):
            if yt == 'UNKNOWN':
                total_unknown += 1
                if yp == 'UNKNOWN':
                    correct_unknown += 1
                else:
                    false_accepts += 1
            else:
                total_known += 1
                if yp == yt:
                    correct_known += 1
                else:
                    false_rejects += 1
                    
        total = total_known + total_unknown
        overall_accuracy = (correct_known + correct_unknown) / total if total > 0 else 0
        
        known_accuracy = correct_known / total_known if total_known > 0 else 0
        unknown_rejection_rate = correct_unknown / total_unknown if total_unknown > 0 else 0
        
        far = false_accepts / total_unknown if total_unknown > 0 else 0
        frr = false_rejects / total_known if total_known > 0 else 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'known_accuracy': known_accuracy,
            'unknown_rejection_rate': unknown_rejection_rate,
            'false_acceptance_rate': far,
            'false_rejection_rate': frr,
            'total_tested': total,
            'total_known': total_known,
            'total_unknown': total_unknown
        }

    def find_optimal_threshold(self, test_dir: str, start: float = 0.50, end: float = 0.90, step: float = 0.05):
        """Sweeps thresholds and prints FAR/FRR to help find the Equal Error Rate (EER)."""
        print(f"\n--- THRESHOLD SWEEP (Calibration) ---")
        print(f"Testing thresholds from {start:.2f} to {end:.2f}...\n")
        print(f"{'Threshold':<12} | {'FAR (False Accepts)':<22} | {'FRR (False Rejects)':<22} | {'Accuracy'}")
        print("-" * 75)
        
        best_threshold = start
        best_accuracy = 0
        min_diff = 1.0 # For EER
        eer_threshold = start
        
        # Generate thresholds safely to avoid floating point issues
        thresholds = np.arange(start, end + (step/2), step)
        
        for t in thresholds:
            metrics = self.evaluate_directory(test_dir, threshold=float(t))
            if 'error' in metrics:
                print("Error:", metrics['error'])
                return
                
            far = metrics['false_acceptance_rate']
            frr = metrics['false_rejection_rate']
            acc = metrics['overall_accuracy']
            
            print(f"{t:<12.2f} | {far:<22.2%} | {frr:<22.2%} | {acc:.2%}")
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = t
                
            if abs(far - frr) < min_diff:
                min_diff = abs(far - frr)
                eer_threshold = t
                
        print("-" * 75)
        print(f"Optimal Threshold (Highest Accuracy): {best_threshold:.2f}")
        print(f"Equal Error Rate (EER) Threshold: {eer_threshold:.2f} (Recommended)")
        print("\nNote: Update 'default_threshold' in recognizer.py once calibrated.")

if __name__ == "__main__":
    import argparse
    from .database import FaceDatabase

    parser = argparse.ArgumentParser(description="Evaluate Face Recognition System")
    parser.add_argument("--test_dir", type=str, default="data/test", help="Path to test directory")
    parser.add_argument("--sweep", action="store_true", help="Run a threshold sweep to find the optimal threshold")
    args = parser.parse_args()

    print("Initializing models (this may take a moment)...")
    detector = FaceDetector(device='cpu')
    embedder = FaceEmbedder(device='cpu')
    db = FaceDatabase()
    recognizer = FaceRecognizer(database=db)
    
    evaluator = FaceEvaluator(detector, embedder, recognizer)
    
    if args.sweep:
        evaluator.find_optimal_threshold(args.test_dir)
    else:
        print(f"\nRunning evaluation with default threshold: {recognizer.default_threshold}")
        metrics = evaluator.evaluate_directory(args.test_dir)
        if 'error' in metrics:
            print(metrics['error'])
        else:
            print(f"Total Tested: {metrics['total_tested']}")
            print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
            print(f"False Acceptance Rate (FAR): {metrics['false_acceptance_rate']:.2%}")
            print(f"False Rejection Rate (FRR): {metrics['false_rejection_rate']:.2%}")

    def find_optimal_threshold(self, test_dir: str, start: float = 0.50, end: float = 0.90, step: float = 0.05):
        """Sweeps thresholds and prints FAR/FRR to help find the Equal Error Rate (EER)."""
        print(f"\n--- THRESHOLD SWEEP (Calibration) ---")
        print(f"Testing thresholds from {start:.2f} to {end:.2f}...\n")
        print(f"{'Threshold':<12} | {'FAR (False Accepts)':<22} | {'FRR (False Rejects)':<22} | {'Accuracy'}")
        print("-" * 75)
        
        best_threshold = start
        best_accuracy = 0
        min_diff = 1.0 # For EER
        eer_threshold = start
        
        # Generate thresholds safely to avoid floating point issues
        thresholds = np.arange(start, end + (step/2), step)
        
        for t in thresholds:
            metrics = self.evaluate_directory(test_dir, threshold=float(t))
            if 'error' in metrics:
                print("Error:", metrics['error'])
                return
                
            far = metrics['false_acceptance_rate']
            frr = metrics['false_rejection_rate']
            acc = metrics['overall_accuracy']
            
            print(f"{t:<12.2f} | {far:<22.2%} | {frr:<22.2%} | {acc:.2%}")
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = t
                
            if abs(far - frr) < min_diff:
                min_diff = abs(far - frr)
                eer_threshold = t
                
        print("-" * 75)
        print(f"Optimal Threshold (Highest Accuracy): {best_threshold:.2f}")
        print(f"Equal Error Rate (EER) Threshold: {eer_threshold:.2f} (Recommended)")
        print("\nNote: Update 'default_threshold' in recognizer.py once calibrated.")

if __name__ == "__main__":
    import argparse
    from .database import FaceDatabase

    parser = argparse.ArgumentParser(description="Evaluate Face Recognition System")
    parser.add_argument("--test_dir", type=str, default="data/test", help="Path to test directory")
    parser.add_argument("--sweep", action="store_true", help="Run a threshold sweep to find the optimal threshold")
    args = parser.parse_args()

    print("Initializing models (this may take a moment)...")
    detector = FaceDetector(device='cpu')
    embedder = FaceEmbedder(device='cpu')
    db = FaceDatabase()
    recognizer = FaceRecognizer(database=db)
    
    evaluator = FaceEvaluator(detector, embedder, recognizer)
    
    if args.sweep:
        evaluator.find_optimal_threshold(args.test_dir)
    else:
        print(f"\nRunning evaluation with default threshold: {recognizer.default_threshold}")
        metrics = evaluator.evaluate_directory(args.test_dir)
        if 'error' in metrics:
            print(metrics['error'])
        else:
            print(f"Total Tested: {metrics['total_tested']}")
            print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
            print(f"False Acceptance Rate (FAR): {metrics['false_acceptance_rate']:.2%}")
            print(f"False Rejection Rate (FRR): {metrics['false_rejection_rate']:.2%}")
