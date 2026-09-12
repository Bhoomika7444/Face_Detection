import cv2
import numpy as np
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
