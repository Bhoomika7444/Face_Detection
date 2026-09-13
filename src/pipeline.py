"""End-to-end pipeline used by the app, the evaluation and the tests.

    image -> load_image -> MTCNN detection -> alignment/crop -> FaceNet embedding
          -> ENROLLMENT: quality/duplicate checks -> store embedding
          -> IDENTIFICATION: cosine similarity vs. all references -> decision policy

Enrollment and identification call the same load/detect/embed code, so preprocessing is
guaranteed to be identical for both.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import config
from .database import FaceDatabase, validate_name, validate_person_id
from .detector import DetectedFace, FaceDetector
from .embedder import FaceEmbedder
from .recognizer import FaceRecognizer, MatchResult, cosine_similarity, validate_thresholds
from .utils import ImageLoadError, ImageSource, face_tensor_to_uint8, load_image


class EnrollmentError(ValueError):
    """The enrollment request itself is invalid (bad ID/name, duplicate identity, no images)."""


class NoFaceError(ValueError):
    pass


class MultipleFacesError(ValueError):
    pass


@dataclass
class FaceObservation:
    face: DetectedFace
    embedding: np.ndarray  # (512,) unit vector


@dataclass
class ImageOutcome:
    label: str
    accepted: bool
    message: str
    face_crop: np.ndarray | None = None  # 160x160 RGB preview of what the model saw (not stored)


@dataclass
class EnrollmentReport:
    person_id: str
    name: str
    outcomes: list[ImageOutcome] = field(default_factory=list)
    stored_count: int = 0
    total_references: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.stored_count > 0


@dataclass
class FaceIdentification:
    index: int                # 1-based, numbered left to right
    face: DetectedFace
    embedding: np.ndarray
    match: MatchResult
    warnings: list[str] = field(default_factory=list)


def quality_issues(face_size: float, prob: float, sharpness: float, brightness: float) -> list[str]:
    """Reasons a face is not good enough to become an enrollment reference (empty = acceptable)."""
    issues = []
    if face_size < config.MIN_ENROLL_FACE_SIZE:
        issues.append(f"face is too small ({face_size:.0f}px; need >= {config.MIN_ENROLL_FACE_SIZE}px). "
                      "Use a closer or higher-resolution photo")
    if prob < config.MIN_ENROLL_DETECTION_PROB:
        issues.append(f"face detection confidence is low ({prob:.2f}); the face may be partly hidden")
    if sharpness < config.MIN_SHARPNESS:
        issues.append(f"face looks blurry (sharpness {sharpness:.0f} < {config.MIN_SHARPNESS:.0f})")
    if not config.MIN_BRIGHTNESS <= brightness <= config.MAX_BRIGHTNESS:
        issues.append(f"face is too dark or overexposed (brightness {brightness:.0f})")
    return issues


def enrollment_quality_issues(face: DetectedFace) -> list[str]:
    q = face.quality
    return quality_issues(q.face_size, face.prob, q.sharpness, q.brightness)


def identification_warnings(face: DetectedFace) -> list[str]:
    q, warnings = face.quality, []
    if q.face_size < config.MIN_RELIABLE_FACE_SIZE:
        warnings.append(f"small face ({q.face_size:.0f}px): result is less reliable")
    if q.sharpness < config.MIN_SHARPNESS:
        warnings.append("blurry face: result is less reliable")
    if not config.MIN_BRIGHTNESS <= q.brightness <= config.MAX_BRIGHTNESS:
        warnings.append("poor lighting: result is less reliable")
    return warnings


class FaceRecognitionSystem:
    def __init__(self, database: FaceDatabase | None = None, detector: FaceDetector | None = None,
                 embedder: FaceEmbedder | None = None, device: str = "cpu"):
        self.database = database if database is not None else FaceDatabase()
        self.detector = detector if detector is not None else FaceDetector(device=device)
        self.embedder = embedder if embedder is not None else FaceEmbedder(device=device)

    # ------------------------------------------------------------------ shared steps
    def extract_faces(self, image: Image.Image) -> list[FaceObservation]:
        """Detect every face and embed all of them in one batch."""
        faces = self.detector.detect_faces(image)
        embeddings = self.embedder.get_embeddings([f.face_tensor for f in faces])
        return [FaceObservation(f, e) for f, e in zip(faces, embeddings)]

    def _recognizer(self, accept: float, uncertain: float) -> FaceRecognizer:
        return FaceRecognizer(self.database, accept, uncertain)

    # ------------------------------------------------------------------ enrollment
    def enroll(self, person_id: str, name: str, images: list[tuple[str, ImageSource | Image.Image]],
               add_to_existing: bool = False,
               accept_threshold: float = config.ACCEPT_THRESHOLD,
               uncertain_threshold: float = config.UNCERTAIN_THRESHOLD) -> EnrollmentReport:
        """Enroll one person from one or more photos. Each photo is checked independently and
        only the accepted ones are stored (as separate reference embeddings)."""
        try:
            person_id, name = validate_person_id(person_id), validate_name(name)
        except ValueError as exc:
            raise EnrollmentError(str(exc)) from exc
        validate_thresholds(accept_threshold, uncertain_threshold)
        if not images:
            raise EnrollmentError("Upload at least one photo of the person.")
        if len(images) > config.MAX_ENROLL_IMAGES_PER_REQUEST:
            raise EnrollmentError(f"At most {config.MAX_ENROLL_IMAGES_PER_REQUEST} photos per enrollment.")

        existing = self.database.get_person(person_id)
        if existing and not add_to_existing:
            raise EnrollmentError(
                f"ID '{person_id}' is already enrolled as '{existing['name']}'. Enable "
                "'Add photos to an existing identity' to add more photos of the same person.")
        if existing and existing["name"].casefold() != name.casefold():
            raise EnrollmentError(f"ID '{person_id}' belongs to '{existing['name']}', not '{name}'.")
        if add_to_existing and not existing:
            raise EnrollmentError(f"ID '{person_id}' is not enrolled yet; untick 'add to existing'.")

        references = [np.asarray(e, dtype=np.float32) for e in existing["embeddings"]] if existing else []
        recognizer = self._recognizer(accept_threshold, uncertain_threshold)
        report = EnrollmentReport(person_id, name)
        accepted: list[np.ndarray] = []

        for label, source in images:
            try:
                image = source if isinstance(source, Image.Image) else load_image(source)
            except ImageLoadError as exc:
                report.outcomes.append(ImageOutcome(label, False, str(exc)))
                continue
            observations = self.extract_faces(image)
            if not observations:
                report.outcomes.append(ImageOutcome(label, False, "No face detected. Use a clear, front-facing photo."))
                continue
            if len(observations) > 1:
                report.outcomes.append(ImageOutcome(
                    label, False, f"{len(observations)} faces detected. Enrollment photos must show only "
                                  "the person being enrolled (we never guess which face is theirs)."))
                continue

            obs = observations[0]
            crop = face_tensor_to_uint8(obs.face.face_tensor)
            issues = enrollment_quality_issues(obs.face)
            if issues:
                report.outcomes.append(ImageOutcome(label, False, "Poor quality: " + "; ".join(issues) + ".", crop))
                continue

            # Compare with this person's other photos (already stored + accepted in this request).
            pool = references + accepted
            if pool:
                best_own = max(cosine_similarity(obs.embedding, r) for r in pool)
                if best_own >= config.DUPLICATE_IMAGE_SIMILARITY:
                    report.outcomes.append(ImageOutcome(
                        label, False, f"Duplicate: this photo is (almost) identical to one already used for "
                                      f"{name} (similarity {best_own:.3f}).", crop))
                    continue
                if best_own < uncertain_threshold:
                    report.outcomes.append(ImageOutcome(
                        label, False, f"This face does not match the other photos of {name} (best similarity "
                                      f"{best_own:.3f} < {uncertain_threshold:.2f}). Check it is the same person.", crop))
                    continue

            # One person must not be enrolled under two IDs: check against everyone else.
            others = [r for r in recognizer.rank_identities(obs.embedding) if r[0] != person_id]
            if others and others[0][2] >= accept_threshold:
                other_id, other_name, other_sim = others[0]
                report.outcomes.append(ImageOutcome(
                    label, False, f"This face already matches enrolled identity '{other_name}' (ID {other_id}, "
                                  f"similarity {other_sim:.3f}). One person should have one ID.", crop))
                continue

            accepted.append(obs.embedding)
            report.outcomes.append(ImageOutcome(
                label, True, f"Accepted (face {obs.face.quality.face_size:.0f}px, detection {obs.face.prob:.3f}).", crop))

        if accepted:
            report.total_references = self.database.add_embeddings(person_id, name, accepted)
            report.stored_count = len(accepted)
            if report.total_references < config.RECOMMENDED_ENROLL_IMAGES:
                report.warnings.append(
                    f"{name} has only {report.total_references} reference photo(s). "
                    f"{config.RECOMMENDED_ENROLL_IMAGES}+ photos with different lighting and pose "
                    "make recognition more reliable.")
        return report

    # ------------------------------------------------------------------ identification
    def identify(self, image: Image.Image, accept_threshold: float = config.ACCEPT_THRESHOLD,
                 uncertain_threshold: float = config.UNCERTAIN_THRESHOLD) -> list[FaceIdentification]:
        """Identify every face in the image independently. Faces are numbered left to right."""
        recognizer = self._recognizer(accept_threshold, uncertain_threshold)
        observations = sorted(self.extract_faces(image), key=lambda o: o.face.box[0])
        return [FaceIdentification(i, o.face, o.embedding, recognizer.identify(o.embedding),
                                   identification_warnings(o.face))
                for i, o in enumerate(observations, start=1)]

    def verify_single(self, image: Image.Image, accept_threshold: float,
                      uncertain_threshold: float) -> FaceIdentification:
        """One re-verification attempt: the photo must contain exactly one face."""
        results = self.identify(image, accept_threshold, uncertain_threshold)
        if not results:
            raise NoFaceError("No face detected in this photo. Please upload a clear photo of the person.")
        if len(results) > 1:
            raise MultipleFacesError(f"{len(results)} faces detected. Re-verification needs a photo "
                                     "containing only the person being verified.")
        return results[0]
