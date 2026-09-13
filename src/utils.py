"""Image loading, face-crop preprocessing, quality measurement and drawing helpers.

Colour convention: inside the pipeline every image is RGB (PIL image or HxWx3 uint8 array).
OpenCV's BGR order is used only for drawing, via pil_to_cv2 / cv2_to_pil.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from . import config


class ImageLoadError(ValueError):
    """The input is not a usable image (corrupted, unsupported format, too small, ...)."""


ImageSource = Union[str, Path, bytes, BinaryIO]


def load_image(source: ImageSource, max_side: int | None = config.MAX_IMAGE_SIDE) -> Image.Image:
    """Load, validate and normalise an image. Returns an RGB PIL image.

    decode -> format check -> EXIF orientation -> RGB -> size checks -> optional downscale.
    Enrollment and identification both call this, so both see identical preprocessing.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    if hasattr(source, "seek"):
        source.seek(0)  # a Streamlit UploadedFile may already have been read
    try:
        img = Image.open(source)
        if img.format not in config.ALLOWED_IMAGE_FORMATS:
            raise ImageLoadError(
                f"Unsupported image format '{img.format}'. "
                f"Allowed formats: {', '.join(sorted(config.ALLOWED_IMAGE_FORMATS))}."
            )
        if img.width * img.height > config.MAX_IMAGE_PIXELS:
            raise ImageLoadError(f"Image is too large ({img.width}x{img.height} pixels).")
        img.load()  # force a full decode so truncated / corrupted files fail here
        # Phone photos store their rotation in EXIF; without this, portrait photos arrive sideways.
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except ImageLoadError:
        raise
    except FileNotFoundError as exc:
        raise ImageLoadError(f"Image file not found: {source}") from exc
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ImageLoadError("The file is not a valid image, or it is corrupted.") from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise ImageLoadError(f"The image could not be decoded (corrupted or truncated): {exc}") from exc

    if min(img.size) < config.MIN_IMAGE_SIDE:
        raise ImageLoadError(
            f"Image is too small ({img.width}x{img.height}); the minimum side is {config.MIN_IMAGE_SIDE} px."
        )
    if max_side is not None and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """PIL RGB image -> OpenCV BGR array."""
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """OpenCV BGR array -> PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def draw_faces(image_bgr: np.ndarray, faces_data: list[dict]) -> np.ndarray:
    """Draw boxes and labels. faces_data: [{'box': [x1,y1,x2,y2], 'label': str, 'color': (B,G,R)}],
    with an optional 'font_scale' per face (default: based on the image size)."""
    img = image_bgr.copy()
    h_img, w_img = img.shape[:2]
    thickness = max(2, round(min(h_img, w_img) / 300))
    font_scale = max(0.5, min(h_img, w_img) / 900)
    for face in faces_data:
        box = face.get("box", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
        color = face.get("color", (0, 0, 255))
        label = str(face.get("label", ""))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            fs = face.get("font_scale", font_scale)
            text_thickness = max(1, round(fs * 1.5))
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, text_thickness)
            # Label above the box, or inside it when the face touches the top edge of the image.
            ty = y1 - 4 if y1 - th - baseline - 4 >= 0 else y1 + th + 4
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), color, -1)
            cv2.putText(img, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), text_thickness, cv2.LINE_AA)
    return img


def _crop_with_zero_padding(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """img[y1:y2, x1:x2], where any part outside the image is filled with zeros (black)."""
    h, w = img.shape[:2]
    if x2 <= x1 or y2 <= y1:
        return img[0:0, 0:0]
    inner = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if inner.size == 0:
        return np.zeros((y2 - y1, x2 - x1, img.shape[2]), dtype=img.dtype)
    return cv2.copyMakeBorder(inner, max(0, -y1), max(0, y2 - h), max(0, -x1), max(0, x2 - w),
                              cv2.BORDER_CONSTANT, value=0)


def align_and_crop(img_rgb: np.ndarray, box: list, landmarks: list, target_size=(160, 160)) -> torch.Tensor:
    """Rotate the face so the eyes are level, crop it, resize to 160x160 and standardise.

    Input is an RGB uint8 array (the model was trained on RGB). Output is a (3, 160, 160)
    float tensor with fixed image standardisation (x - 127.5) / 128, the normalisation used to
    train the facenet-pytorch VGGFace2 model. A preprocessing ablation on 1,200 official LFW
    pairs showed this crop performs the same as facenet-pytorch's own crop (see README).
    """
    x1, y1, x2, y2 = [int(v) for v in box]

    if len(landmarks) >= 5:
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        # Angle of the eye line relative to horizontal
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Rotate only a padded patch around the face instead of the whole image (faster).
        # Parts of the patch outside the image are zero-filled instead of clipped, so a face
        # touching the image border is never stretched when resized to 160x160.
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * 0.5), int(bh * 0.5)

        px1, py1 = x1 - pad_x, y1 - pad_y
        px2, py2 = x2 + pad_x, y2 + pad_y

        patch = _crop_with_zero_padding(img_rgb, px1, py1, px2, py2)

        # Landmarks in patch coordinates
        patch_left_eye = (left_eye[0] - px1, left_eye[1] - py1)
        patch_right_eye = (right_eye[0] - px1, right_eye[1] - py1)

        # Rotate around the midpoint between the eyes
        eye_center = (
            int((patch_left_eye[0] + patch_right_eye[0]) // 2),
            int((patch_left_eye[1] + patch_right_eye[1]) // 2)
        )

        M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
        ph, pw = patch.shape[:2]
        aligned_patch = cv2.warpAffine(patch, M, (pw, ph), flags=cv2.INTER_LINEAR)

        # Rotate the box corners with the same transform and crop their bounding rectangle
        patch_x1, patch_y1 = x1 - px1, y1 - py1
        patch_x2, patch_y2 = x2 - px1, y2 - py1

        box_pts = np.array([
            [patch_x1, patch_y1], [patch_x2, patch_y1],
            [patch_x2, patch_y2], [patch_x1, patch_y2]
        ])
        ones = np.ones(shape=(len(box_pts), 1))
        points_ones = np.hstack([box_pts, ones])
        transformed_points = M.dot(points_ones.T).T

        x_coords = transformed_points[:, 0]
        y_coords = transformed_points[:, 1]

        new_x1, new_y1 = int(np.min(x_coords)), int(np.min(y_coords))
        new_x2, new_y2 = int(np.max(x_coords)), int(np.max(y_coords))

        new_x1, new_y1 = max(0, new_x1), max(0, new_y1)
        new_x2, new_y2 = min(pw, new_x2), min(ph, new_y2)

        cropped_img = aligned_patch[new_y1:new_y2, new_x1:new_x2]
    else:
        # Fallback to a plain box crop if landmarks are missing
        cropped_img = _crop_with_zero_padding(img_rgb, x1, y1, x2, y2)

    if cropped_img.size == 0 or cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
        cropped_img = np.zeros((*target_size, 3), dtype=np.uint8)
    else:
        cropped_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)

    # Fixed image standardisation (the normalisation the VGGFace2 FaceNet model expects)
    cropped_float = (cropped_img.astype(np.float32) - 127.5) / 128.0
    return torch.from_numpy(cropped_float).permute(2, 0, 1).float()


def face_tensor_to_uint8(face_tensor: torch.Tensor) -> np.ndarray:
    """Undo the standardisation: (3,160,160) tensor -> 160x160x3 RGB uint8 (what the model sees)."""
    arr = face_tensor.detach().cpu().permute(1, 2, 0).numpy() * 128.0 + 127.5
    return np.clip(arr, 0, 255).astype(np.uint8)


@dataclass
class FaceQuality:
    """Simple, explainable quality measurements of one detected face."""
    face_size: float   # shorter side of the detection box, in pixels
    sharpness: float   # variance of the Laplacian of the grey 160x160 crop (low = blurry)
    brightness: float  # mean grey level of the crop, 0-255


def measure_face_quality(face_tensor: torch.Tensor, box: list) -> FaceQuality:
    grey = cv2.cvtColor(face_tensor_to_uint8(face_tensor), cv2.COLOR_RGB2GRAY)
    return FaceQuality(
        face_size=float(min(box[2] - box[0], box[3] - box[1])),
        sharpness=float(cv2.Laplacian(grey, cv2.CV_64F).var()),
        brightness=float(grey.mean()),
    )
