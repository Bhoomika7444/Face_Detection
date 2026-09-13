"""Face detection with MTCNN (facenet-pytorch).

Detection answers "where are the faces?". It returns boxes, 5 facial landmarks and an aligned
160x160 crop per face. It does NOT say who the person is; that is the embedder + recognizer.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from . import config
from .utils import FaceQuality, align_and_crop, measure_face_quality

warnings.filterwarnings("ignore", category=UserWarning, module="facenet_pytorch")


class ModelInitError(RuntimeError):
    """A pretrained model could not be loaded."""


class FaceDetectionError(RuntimeError):
    """The detector failed on an input image."""


@dataclass
class DetectedFace:
    box: list[float]              # [x1, y1, x2, y2] in image pixels
    prob: float                   # MTCNN detection probability
    landmarks: list[list[float]]  # left eye, right eye, nose, mouth left, mouth right
    face_tensor: torch.Tensor     # (3, 160, 160) aligned, standardised RGB crop for the embedder
    quality: FaceQuality

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available; falling back to CPU.")
        device = "cpu"
    return torch.device(device)


class FaceDetector:
    def __init__(self, device: str = "cpu", min_prob: float = config.MIN_DETECTION_PROB):
        """MTCNN = cascade of 3 small CNNs (P-Net, R-Net, O-Net). Weights ship inside facenet-pytorch."""
        from facenet_pytorch import MTCNN

        self.device = resolve_device(device)
        self.min_prob = min_prob
        try:
            self.mtcnn = MTCNN(
                keep_all=True,  # return every face, not just one
                min_face_size=config.MTCNN_MIN_FACE_SIZE,
                thresholds=list(config.MTCNN_STAGE_THRESHOLDS),
                device=self.device,
            )
        except Exception as exc:  # re-raised with a clear message, never swallowed
            raise ModelInitError(f"Could not initialise the MTCNN face detector: {exc}") from exc

    def detect_faces(self, image: Image.Image) -> list[DetectedFace]:
        """Detect all faces with probability >= min_prob. Largest face first (MTCNN order)."""
        if not isinstance(image, Image.Image):
            raise TypeError("detect_faces expects a PIL image (use utils.load_image).")
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        try:
            boxes, probs, landmarks = self.mtcnn.detect(rgb, landmarks=True)
        except Exception as exc:
            raise FaceDetectionError(f"Face detection failed on this image: {exc}") from exc
        if boxes is None:
            return []

        img_rgb = np.asarray(rgb)
        faces = []
        for box, prob, lm in zip(boxes, probs, landmarks):
            if prob is None or prob < self.min_prob:
                continue
            b, points = box.tolist(), lm.tolist()
            face_tensor = align_and_crop(img_rgb, b, points, target_size=(config.FACE_CROP_SIZE,) * 2)
            faces.append(DetectedFace(
                box=b,
                prob=float(prob),
                landmarks=points,
                face_tensor=face_tensor,
                quality=measure_face_quality(face_tensor, b),
            ))
        return faces

    def crop_face(self, image: Image.Image, box: list) -> Image.Image:
        """Plain (unaligned) crop of a box, for display."""
        x1, y1, x2, y2 = [int(b) for b in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        return image.crop((x1, y1, x2, y2))


def select_central_face(faces: list[DetectedFace], image_size: tuple[int, int]) -> DetectedFace | None:
    """Pick the face closest to the image centre.

    Used only by the evaluation code for datasets whose label refers to the centred person
    (LFW images are centred on the labelled person; picking the *largest* face chose someone
    else in 54 of 1,583 LFW images we checked). The app never guesses: enrollment rejects
    multi-face photos and identification reports every face.
    """
    if not faces:
        return None
    cx, cy = image_size[0] / 2, image_size[1] / 2
    return min(faces, key=lambda f: (f.center[0] - cx) ** 2 + (f.center[1] - cy) ** 2)
