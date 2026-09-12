import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import warnings
from .utils import align_and_crop

# Suppress PyTorch warnings for clean output
warnings.filterwarnings('ignore', category=UserWarning)

class FaceDetector:
    def __init__(self, device='cpu'):
        """
        Initializes the face detector using MTCNN.
        MTCNN handles face detection and alignment.
        """
        # Determine device
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
            
        self.device = torch.device(device)
        
        # Initialize MTCNN
        # keep_all=True allows detecting multiple faces in one image
        # min_face_size=20 is standard
        # thresholds=[0.6, 0.7, 0.7] are standard MTCNN thresholds
        self.mtcnn = MTCNN(
            keep_all=True, 
            min_face_size=20, 
            thresholds=[0.6, 0.7, 0.7], 
            device=self.device
        )

    def detect_faces(self, image: Image.Image) -> list:
        """
        Detects faces in a PIL Image.
        Returns a list of dictionaries with keys:
            - 'box': [x1, y1, x2, y2]
            - 'prob': probability score
            - 'landmarks': facial landmarks (optional)
            - 'face_tensor': cropped and pre-whitened face tensor ready for FaceNet
        """
        # MTCNN returns bounding boxes, probabilities, and landmarks
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        
        # If no faces detected, return empty list
        if boxes is None:
            return []
            
        faces_data = []
        
        # Process each detected face strictly using the box and landmarks
        for i, box in enumerate(boxes):
            if probs[i] is None or probs[i] < 0.90:
                continue # Ignore low confidence faces
                
            lm = landmarks[i].tolist() if landmarks is not None else []
            b = box.tolist()
            
            # Use our custom alignment and extraction pipeline
            face_tensor = align_and_crop(image, b, lm, target_size=(160, 160))
            
            faces_data.append({
                'box': b,
                'prob': float(probs[i]),
                'landmarks': lm,
                'face_tensor': face_tensor # 3x160x160 aligned and pre-whitened tensor
            })
                
        return faces_data
        
    def crop_face(self, image: Image.Image, box: list) -> Image.Image:
        """
        Helper method to manually crop a face given a bounding box.
        """
        x1, y1, x2, y2 = [int(b) for b in box]
        # Add a little padding optionally
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        return image.crop((x1, y1, x2, y2))
