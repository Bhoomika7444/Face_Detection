import cv2
import numpy as np
import torch
from PIL import Image

def load_image(image_path: str) -> Image.Image:
    """Loads an image from path as a PIL Image."""
    return Image.open(image_path).convert('RGB')

def load_image_cv2(image_path: str) -> np.ndarray:
    """Loads an image from path as an OpenCV BGR array."""
    return cv2.imread(image_path)

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Converts a PIL Image (RGB) to OpenCV BGR."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Converts an OpenCV BGR image to PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def draw_faces(image_cv2: np.ndarray, faces_data: list) -> np.ndarray:
    """
    Draws bounding boxes and labels on an image.
    faces_data is a list of dicts:
    [{'box': [x1, y1, x2, y2], 'label': 'Name', 'score': 0.85, 'color': (0, 255, 0)}]
    """
    img_copy = image_cv2.copy()
    for face in faces_data:
        box = face.get('box', [])
        if len(box) == 4:
            x1, y1, x2, y2 = [int(v) for v in box]
            label = face.get('label', 'Unknown')
            score = face.get('score', 0.0)
            color = face.get('color', (0, 0, 255)) # Default to red (BGR)
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            text = f"{label} ({score:.2f})" if score > 0 else label
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_copy, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # Draw label text
            cv2.putText(img_copy, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
    return img_copy

def align_and_crop(image: Image.Image, box: list, landmarks: list, target_size=(160, 160)) -> torch.Tensor:
    """
    Performs affine alignment using MTCNN landmarks, crops the face, and 
    returns a pre-whitened PyTorch tensor (C, H, W) for FaceNet.
    """
    img_cv2 = np.array(image.convert('RGB'))
    x1, y1, x2, y2 = [int(v) for v in box]
    
    if len(landmarks) >= 5:
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # Calculate angle to horizontal
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotate around the center between the eyes
        eye_center = (
            int((left_eye[0] + right_eye[0]) // 2),
            int((left_eye[1] + right_eye[1]) // 2)
        )
        
        M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
        h, w = img_cv2.shape[:2]
        aligned_img = cv2.warpAffine(img_cv2, M, (w, h), flags=cv2.INTER_CUBIC)
        
        # Transform the bounding box coordinates
        box_pts = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ])
        ones = np.ones(shape=(len(box_pts), 1))
        points_ones = np.hstack([box_pts, ones])
        transformed_points = M.dot(points_ones.T).T
        
        x_coords = transformed_points[:, 0]
        y_coords = transformed_points[:, 1]
        
        new_x1, new_y1 = int(np.min(x_coords)), int(np.min(y_coords))
        new_x2, new_y2 = int(np.max(x_coords)), int(np.max(y_coords))
        
        new_x1, new_y1 = max(0, new_x1), max(0, new_y1)
        new_x2, new_y2 = min(w, new_x2), min(h, new_y2)
        
        cropped_img = aligned_img[new_y1:new_y2, new_x1:new_x2]
    else:
        # Fallback to standard crop if landmarks missing
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_cv2.shape[1], x2), min(img_cv2.shape[0], y2)
        cropped_img = img_cv2[y1:y2, x1:x2]
        
    # Handle empty crops securely
    if cropped_img.size == 0 or cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
        cropped_img = np.zeros((*target_size, 3), dtype=np.uint8)
    else:
        cropped_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)
        
    # Pre-whiten (standard FaceNet format)
    cropped_float = cropped_img.astype(np.float32)
    cropped_float = (cropped_float - 127.5) / 128.0
    
    # Convert to tensor (C, H, W)
    tensor = torch.from_numpy(cropped_float).permute(2, 0, 1).float()
    return tensor

