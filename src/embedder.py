"""Face embeddings with a pretrained FaceNet model (InceptionResnetV1, VGGFace2 weights).

An embedding is a 512-number vector that describes a face. The network was trained so that
photos of the same person give vectors pointing in similar directions (high cosine
similarity) and different people give dissimilar vectors. We only run inference; no training.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from . import config
from .detector import ModelInitError, resolve_device


class EmbeddingError(RuntimeError):
    """The model produced an unusable embedding (NaN/inf/zero vector)."""


class FaceEmbedder:
    def __init__(self, device: str = "cpu"):
        from facenet_pytorch import InceptionResnetV1

        self.device = resolve_device(device)
        try:
            # First run downloads ~107 MB of weights into the torch cache (~/.cache/torch/checkpoints).
            self.model = InceptionResnetV1(pretrained="vggface2", device=self.device).eval()
        except Exception as exc:
            raise ModelInitError(
                "Could not load the pretrained FaceNet (VGGFace2) weights. On first run they are "
                "downloaded from the facenet-pytorch GitHub releases into ~/.cache/torch/checkpoints; "
                f"check your internet connection and disk space. Original error: {exc}"
            ) from exc

    def get_embeddings(self, face_tensors: Sequence[torch.Tensor]) -> np.ndarray:
        """Embed a batch of (3,160,160) face tensors -> (n, 512) float32, each row L2-normalised."""
        if len(face_tensors) == 0:
            return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)
        batch = torch.stack([t.squeeze(0) if t.dim() == 4 else t for t in face_tensors])
        expected = (3, config.FACE_CROP_SIZE, config.FACE_CROP_SIZE)
        if tuple(batch.shape[1:]) != expected:
            raise ValueError(f"Face tensors must have shape {expected}, got {tuple(batch.shape[1:])}.")
        with torch.no_grad():  # inference only
            out = self.model(batch.to(self.device)).cpu().numpy().astype(np.float32)

        norms = np.linalg.norm(out, axis=1, keepdims=True)
        if not np.all(np.isfinite(out)) or np.any(norms == 0):
            raise EmbeddingError("The model returned an invalid embedding (NaN/inf or zero vector).")
        # facenet-pytorch already L2-normalises; normalising again is cheap and makes it explicit.
        return out / norms

    def get_embedding(self, face_tensor: torch.Tensor) -> np.ndarray:
        """Embed one face -> (512,) L2-normalised vector."""
        return self.get_embeddings([face_tensor])[0]
