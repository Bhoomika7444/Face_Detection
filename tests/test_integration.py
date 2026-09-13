"""Integration tests: the REAL pretrained models on REAL photos (LFW).

They are skipped (with a reason) when LFW is not on disk. To enable them:
    python scripts/prepare_lfw.py          # downloads LFW to ~/scikit_learn_data/lfw_home
or set LFW_DIR to a folder containing LFW's <Person_Name>/<Person_Name>_0001.jpg files.

Run only these:   pytest -m integration
Skip them:        pytest -m "not integration"
"""
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.database import FaceDatabase
from src.detector import FaceDetector, ModelInitError
from src.embedder import FaceEmbedder
from src.pipeline import FaceRecognitionSystem, MultipleFacesError, NoFaceError
from src.recognizer import RECOGNIZED, UNCERTAIN, UNKNOWN, cosine_similarity
from src.utils import load_image
from src.verification import VerificationSession

LFW = Path(os.environ.get("LFW_DIR", Path.home() / "scikit_learn_data" / "lfw_home" / "lfw"))
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not LFW.is_dir(), reason=f"LFW not found at {LFW}; run scripts/prepare_lfw.py")]


def lfw(name: str, i: int) -> Path:
    return LFW / name / f"{name}_{i:04d}.jpg"


@pytest.fixture(scope="module")
def models():
    try:
        return FaceDetector(), FaceEmbedder()
    except ModelInitError as exc:
        pytest.skip(f"pretrained weights unavailable: {exc}")


@pytest.fixture
def system(models, tmp_path):
    return FaceRecognitionSystem(FaceDatabase(db_dir=tmp_path), *models)


def embed(models, path: Path) -> np.ndarray:
    det, emb = models
    faces = det.detect_faces(load_image(path))
    assert len(faces) == 1, f"expected one face in {path.name}"
    return emb.get_embedding(faces[0].face_tensor)


def composite(paths) -> Image.Image:
    canvas = Image.new("RGB", (250 * len(paths), 250))
    for i, p in enumerate(paths):
        canvas.paste(load_image(p).resize((250, 250)), (250 * i, 0))
    return canvas


# ---------------------------------------------------------------- detector
def test_detector_finds_single_face(models):
    img = load_image(lfw("Colin_Powell", 1))
    faces = models[0].detect_faces(img)
    assert len(faces) == 1
    f = faces[0]
    assert f.prob > 0.99 and len(f.landmarks) == 5
    assert 0 <= f.box[0] < f.box[2] <= img.width and 0 <= f.box[1] < f.box[3] <= img.height
    assert tuple(f.face_tensor.shape) == (3, 160, 160)


def test_detector_no_face_in_blank_or_noise(models):
    assert models[0].detect_faces(Image.new("RGB", (300, 300), "white")) == []
    noise = np.random.default_rng(0).integers(0, 256, (300, 300, 3), dtype=np.uint8)
    assert models[0].detect_faces(Image.fromarray(noise)) == []


def test_detector_finds_every_face_in_a_group_image(models):
    img = composite([lfw("Colin_Powell", 20), lfw("Serena_Williams", 1), lfw("Tony_Blair", 1), lfw("Gloria_Macapagal_Arroyo", 1)])
    faces = models[0].detect_faces(img)
    tiles = {int(f.center[0] // 250) for f in faces}
    assert len(faces) >= 4 and tiles == {0, 1, 2, 3}


# ---------------------------------------------------------------- embedder
def test_embedding_shape_norm_and_determinism(models):
    det, emb = models
    face = det.detect_faces(load_image(lfw("Colin_Powell", 1)))[0]
    e1, e2 = emb.get_embedding(face.face_tensor), emb.get_embedding(face.face_tensor)
    assert e1.shape == (512,) and np.linalg.norm(e1) == pytest.approx(1.0, abs=1e-5)
    # Deterministic up to float rounding (multi-threaded CPU maths may differ in the last bits).
    assert np.allclose(e1, e2, atol=1e-6)
    batch = emb.get_embeddings([face.face_tensor, face.face_tensor])
    assert batch.shape == (2, 512) and np.allclose(batch[0], e1, atol=1e-5)


def test_same_person_scores_higher_than_different_people(models):
    p1, p2 = embed(models, lfw("Colin_Powell", 1)), embed(models, lfw("Colin_Powell", 2))
    s1, s2 = embed(models, lfw("Serena_Williams", 1)), embed(models, lfw("Serena_Williams", 2))
    genuine = min(cosine_similarity(p1, p2), cosine_similarity(s1, s2))
    impostor = max(cosine_similarity(p1, s1), cosine_similarity(p2, s2))
    assert genuine > impostor + 0.3


# ---------------------------------------------------------------- enrollment + identification
def test_enroll_identify_and_reject_unknown(system):
    report = system.enroll("cpowell", "Colin Powell", [(f"p{i}", lfw("Colin_Powell", i)) for i in (1, 2, 3)])
    assert report.stored_count == 3 and system.database.num_embeddings == 3

    known = system.identify(load_image(lfw("Colin_Powell", 20)))
    assert len(known) == 1 and known[0].match.decision == RECOGNIZED and known[0].match.candidate_id == "cpowell"

    stranger = system.identify(load_image(lfw("Serena_Williams", 1)))
    assert len(stranger) == 1 and stranger[0].match.decision == UNKNOWN
    assert stranger[0].match.candidate_id == "cpowell"  # closest enrolled person, correctly NOT accepted


def test_enrollment_rejects_duplicates_multi_face_and_wrong_person(system):
    system.enroll("cpowell", "Colin Powell", [("p1", lfw("Colin_Powell", 1))])
    report = system.enroll("cpowell", "Colin Powell", [
        ("same photo again", lfw("Colin_Powell", 1)),
        ("group photo", lfw("Colin_Powell", 10)),        # contains 3 faces
        ("someone else", lfw("Tony_Blair", 1)),
    ], add_to_existing=True)
    msgs = {o.label: o.message for o in report.outcomes}
    assert not report.success
    assert "Duplicate" in msgs["same photo again"]
    assert "faces detected" in msgs["group photo"]
    assert "does not match the other photos" in msgs["someone else"]


def test_database_persists_real_embeddings(models, tmp_path):
    FaceRecognitionSystem(FaceDatabase(db_dir=tmp_path), *models).enroll(
        "cpowell", "Colin Powell", [(f"p{i}", lfw("Colin_Powell", i)) for i in (1, 2, 3)])
    reloaded = FaceRecognitionSystem(FaceDatabase(db_dir=tmp_path), *models)  # simulates an app restart
    assert reloaded.identify(load_image(lfw("Colin_Powell", 20)))[0].match.decision == RECOGNIZED


def test_group_image_only_enrolled_person_is_recognized(system):
    """Critical false-acceptance check: A enrolled; B, C, D in the same image are not."""
    system.enroll("cpowell", "Colin Powell", [(f"p{i}", lfw("Colin_Powell", i)) for i in (1, 2, 3)])
    tiles = [lfw("Serena_Williams", 1), lfw("Colin_Powell", 20), lfw("Tony_Blair", 1), lfw("Gloria_Macapagal_Arroyo", 1)]
    results = system.identify(composite(tiles))
    by_tile = {}
    for r in results:  # the face nearest each tile's centre is that tile's person
        t = int(r.face.center[0] // 250)
        d = abs(r.face.center[0] - (250 * t + 125))
        if t not in by_tile or d < by_tile[t][0]:
            by_tile[t] = (d, r)
    assert set(by_tile) == {0, 1, 2, 3}
    assert by_tile[1][1].match.decision == RECOGNIZED and by_tile[1][1].match.candidate_id == "cpowell"
    for t in (0, 2, 3):
        assert by_tile[t][1].match.decision != RECOGNIZED
    assert sum(r.match.decision == RECOGNIZED for r in results) == 1


# ---------------------------------------------------------------- re-verification with the real pipeline
def test_reverification_flow_with_real_scores(system):
    system.enroll("cpowell", "Colin Powell", [(f"p{i}", lfw("Colin_Powell", i)) for i in (1, 2, 3)])
    probe = load_image(lfw("Colin_Powell", 20))
    real_score = system.identify(probe)[0].match.similarity
    # Put the operating point around this photo's real score so that it is borderline.
    accept, uncertain = round(real_score + 0.05, 4), round(real_score - 0.05, 4)
    session = VerificationSession(accept, uncertain, max_attempts=3)

    first = system.verify_single(probe, accept, uncertain)
    assert first.match.decision == UNCERTAIN and session.record(first.match) is None
    same_again = system.verify_single(probe, accept, uncertain)  # re-uploading the same photo
    assert same_again.match.similarity == pytest.approx(first.match.similarity, abs=1e-5)
    assert session.record(same_again.match) is None and session.next_attempt_number == 3
    better = system.verify_single(load_image(lfw("Colin_Powell", 2)), accept, uncertain)  # enrolled photo
    assert session.record(better.match) == RECOGNIZED

    with pytest.raises(NoFaceError):
        system.verify_single(Image.new("RGB", (300, 300), "white"), accept, uncertain)
    with pytest.raises(MultipleFacesError):
        system.verify_single(load_image(lfw("Colin_Powell", 10)), accept, uncertain)
