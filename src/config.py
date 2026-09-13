"""Central configuration.

Every tunable number lives here so that the Streamlit app, the evaluation code and the
tests all use exactly the same values. The UI lets you override the decision thresholds
at runtime; the values below are the defaults.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Optional override, e.g. to keep a separate database for testing or deployment.
DEFAULT_DB_PATH = Path(os.environ.get("FACE_DB_PATH", PROJECT_ROOT / "embeddings" / "database.json"))
RESULTS_DIR = PROJECT_ROOT / "results"

# --------------------------------------------------------------------------------------
# Model (facenet-pytorch). Changing the model makes stored embeddings incompatible, so the
# name is written into the database file and checked on load.
# --------------------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "facenet-pytorch/InceptionResnetV1/vggface2"
EMBEDDING_DIM = 512
FACE_CROP_SIZE = 160  # InceptionResnetV1 input is 160x160 RGB

# --------------------------------------------------------------------------------------
# Face detection (MTCNN)
# --------------------------------------------------------------------------------------
MTCNN_MIN_FACE_SIZE = 20                  # px; smallest face the detector searches for
MTCNN_STAGE_THRESHOLDS = (0.6, 0.7, 0.7)  # standard P-Net / R-Net / O-Net thresholds
MIN_DETECTION_PROB = 0.90                 # discard weaker detections (mostly false positives)

# --------------------------------------------------------------------------------------
# Decision policy on the cosine similarity of the best-matching enrolled identity:
#   similarity >= ACCEPT_THRESHOLD                        -> RECOGNIZED
#   UNCERTAIN_THRESHOLD <= similarity < ACCEPT_THRESHOLD  -> UNCERTAIN (ask for another photo)
#   similarity < UNCERTAIN_THRESHOLD                      -> UNKNOWN
# Values chosen on the LFW *validation* split (results/lfw_val/calibration.json) with the rule
# "FAR <= 1% for a single photo AND after up to 3 re-verification attempts; at most 2% of
# enrolled people rejected outright", then checked on a separate test split with different
# people (results/lfw_test/report.md). They are sensible starting values for a gallery of
# ~150 people, NOT universally optimal: re-calibrate on your own photos (README section 16).
# --------------------------------------------------------------------------------------
ACCEPT_THRESHOLD = 0.71
UNCERTAIN_THRESHOLD = 0.61
MAX_VERIFICATION_ATTEMPTS = 3

# --------------------------------------------------------------------------------------
# Enrollment quality gates (heuristics; see README "Enrollment")
# --------------------------------------------------------------------------------------
MIN_ENROLL_FACE_SIZE = 60          # px, shorter side of the detected face box
MIN_ENROLL_DETECTION_PROB = 0.95   # stricter than identification: references must be clean
MIN_SHARPNESS = 20.0               # variance of Laplacian on the 160x160 grey crop
MIN_BRIGHTNESS = 40.0              # mean grey level (0-255) of the face crop
MAX_BRIGHTNESS = 225.0
DUPLICATE_IMAGE_SIMILARITY = 0.97  # >= this vs. an existing reference => same photo again
RECOMMENDED_ENROLL_IMAGES = 3
MAX_ENROLL_IMAGES_PER_REQUEST = 10

# Identification: faces smaller than this still get a result, but with a warning.
MIN_RELIABLE_FACE_SIZE = 40

# --------------------------------------------------------------------------------------
# Image input
# --------------------------------------------------------------------------------------
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "BMP", "WEBP", "MPO"}  # MPO = multi-picture JPEG from phones
ALLOWED_UPLOAD_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp"]
MIN_IMAGE_SIDE = 32         # px
MAX_IMAGE_PIXELS = 50_000_000
MAX_IMAGE_SIDE = 1600       # larger photos are downscaled (keeps MTCNN fast on CPU)
