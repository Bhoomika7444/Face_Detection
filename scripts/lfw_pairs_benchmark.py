"""Pipeline sanity check: the standard LFW face-verification benchmark (6,000 official pairs).

    python scripts/lfw_pairs_benchmark.py

Protocol (LFW "unrestricted, labeled outside data"): pairs.txt holds 10 folds of 300 same-person
and 300 different-person pairs. For each fold, the threshold with the best accuracy on the other
9 folds is applied to the held-out fold; we report mean +/- std accuracy over the 10 folds.

facenet-pytorch reports ~99.6% for this model. Reproducing that number shows that detection,
alignment, RGB handling, normalisation and cosine similarity are all implemented correctly.
This is 1:1 verification; the identification thresholds (1:N) are calibrated separately by
src/evaluation.py, because the best of N impostor scores is higher than a single one.
"""
from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_lfw import DEFAULT_LFW_HOME, ensure_lfw, sha256  # noqa: E402
from src import config  # noqa: E402
from src.database import FaceDatabase  # noqa: E402
from src.evaluation import embed_images  # noqa: E402
from src.pipeline import FaceRecognitionSystem  # noqa: E402

PAIRS_URL = "https://ndownloader.figshare.com/files/5976006"
PAIRS_SHA256 = "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"


def load_pairs(lfw_home: Path, images: Path):
    pairs_file = lfw_home / "pairs.txt"
    if not pairs_file.exists():
        urllib.request.urlretrieve(PAIRS_URL, pairs_file)
    if sha256(pairs_file) != PAIRS_SHA256:
        raise SystemExit(f"Checksum mismatch for {pairs_file}")
    lines = pairs_file.read_text().strip().splitlines()
    n_folds, n_per = map(int, lines[0].split())
    pairs = []
    for line in lines[1:]:
        p = line.split("\t")
        if len(p) == 3:
            a, b, same = (p[0], int(p[1])), (p[0], int(p[2])), 1
        else:
            a, b, same = (p[0], int(p[1])), (p[2], int(p[3])), 0
        pairs.append((images / a[0] / f"{a[0]}_{a[1]:04d}.jpg", images / b[0] / f"{b[0]}_{b[1]:04d}.jpg", same))
    return pairs, n_folds, 2 * n_per


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve = probability a random same-person pair outscores a different-person pair."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def main() -> None:
    images = ensure_lfw(DEFAULT_LFW_HOME)
    pairs, n_folds, fold_size = load_pairs(DEFAULT_LFW_HOME, images)
    paths = sorted({p for a, b, _ in pairs for p in (a, b)})
    print(f"{len(pairs)} pairs, {len(paths)} unique images")
    with tempfile.TemporaryDirectory() as tmp:
        system = FaceRecognitionSystem(FaceDatabase(db_dir=tmp))
        records = embed_images(system, paths, config.RESULTS_DIR / "cache")

    scores, labels, missing = [], [], 0
    for a, b, same in pairs:
        ra, rb = records[a], records[b]
        if ra.embedding is None or rb.embedding is None:
            missing += 1
            s = -1.0  # no face -> counted as "different" (a failure for same-person pairs)
        else:
            s = float(np.dot(ra.embedding, rb.embedding))  # unit vectors: dot product = cosine similarity
        scores.append(s)
        labels.append(same)
    scores, labels = np.array(scores), np.array(labels)

    grid = np.arange(-0.2, 1.0, 0.005)
    accs, ths = [], []
    for k in range(n_folds):
        test = np.zeros(len(pairs), bool)
        test[k * fold_size:(k + 1) * fold_size] = True
        train_acc = [np.mean((scores[~test] >= t) == labels[~test]) for t in grid]
        t = grid[int(np.argmax(train_acc))]
        ths.append(float(t))
        accs.append(float(np.mean((scores[test] >= t) == labels[test])))

    impostor = np.sort(scores[labels == 0])[::-1]
    def tar_at(far):
        t = impostor[max(int(np.floor(far * len(impostor))) - 1, 0)]
        return float(np.mean(scores[labels == 1] > t))

    summary = {
        "benchmark": "LFW verification, official 10-fold pairs.txt (6,000 pairs)",
        "pairs": len(pairs), "pairs_with_undetected_face": missing,
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "fold_thresholds": ths, "auc": auc(labels, scores),
        "tar_at_far_1e-2": tar_at(1e-2), "tar_at_far_1e-3": tar_at(1e-3),
        "same_person_mean_similarity": float(scores[labels == 1].mean()),
        "different_person_mean_similarity": float(scores[labels == 0].mean()),
        "reference": "facenet-pytorch README reports 0.9965 for this model",
    }
    out = config.RESULTS_DIR / "lfw_pairs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
