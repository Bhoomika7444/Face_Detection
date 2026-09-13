"""Unit tests: pure logic, no model weights or datasets needed (fast, deterministic).

Run: pytest tests/test_core.py
"""
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

from src import config
from src.database import DatabaseError, FaceDatabase, backup_corrupt_database, validate_name, validate_person_id
from src.detector import DetectedFace
from src.pipeline import EnrollmentError, FaceRecognitionSystem, MultipleFacesError, NoFaceError
from src.recognizer import (RECOGNIZED, UNCERTAIN, UNKNOWN, FaceRecognizer, cosine_similarity, decide,
                            validate_thresholds)
from src.utils import FaceQuality, ImageLoadError, _crop_with_zero_padding, draw_faces, load_image
from src.verification import UNABLE_TO_VERIFY, VerificationSession, VerificationSessionClosed

DIM = config.EMBEDDING_DIM


def unit(i: int) -> np.ndarray:
    v = np.zeros(DIM)
    v[i] = 1.0
    return v


def with_similarity(query_axis: int, other_axis: int, sim: float) -> np.ndarray:
    """A unit vector whose cosine similarity with unit(query_axis) is exactly `sim`."""
    return sim * unit(query_axis) + np.sqrt(1 - sim ** 2) * unit(other_axis)


@pytest.fixture
def db(tmp_path):
    return FaceDatabase(db_dir=tmp_path)


# ---------------------------------------------------------------- cosine similarity
def test_cosine_similarity_basic_values():
    a, b = unit(0), unit(1)
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, b) == pytest.approx(0.0)
    assert cosine_similarity(a, -a) == pytest.approx(-1.0)


def test_cosine_similarity_is_scale_invariant_and_matches_formula():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=DIM), rng.normal(size=DIM)
    expected = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    assert cosine_similarity(a, b) == pytest.approx(expected)
    assert cosine_similarity(5 * a, 0.1 * b) == pytest.approx(expected)


def test_cosine_similarity_zero_vector_and_shape_mismatch():
    assert cosine_similarity(np.zeros(DIM), unit(0)) == 0.0
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(3), np.ones(4))


# ---------------------------------------------------------------- decision policy
@pytest.mark.parametrize("sim,expected", [
    (0.95, RECOGNIZED), (0.70, RECOGNIZED),       # at the accept threshold counts as accepted
    (0.69, UNCERTAIN), (0.65, UNCERTAIN), (0.60, UNCERTAIN),
    (0.59, UNKNOWN), (0.10, UNKNOWN), (-0.5, UNKNOWN),
])
def test_decide_three_states(sim, expected):
    assert decide(sim, accept_threshold=0.70, uncertain_threshold=0.60) == expected


def test_decide_band_disabled_when_thresholds_equal():
    assert decide(0.69, 0.70, 0.70) == UNKNOWN
    assert decide(0.70, 0.70, 0.70) == RECOGNIZED


@pytest.mark.parametrize("accept,uncertain", [(0.5, 0.6), (1.5, 0.5), (0.5, -2.0)])
def test_invalid_thresholds_rejected(accept, uncertain):
    with pytest.raises(ValueError):
        validate_thresholds(accept, uncertain)


# ---------------------------------------------------------------- recognizer
def test_empty_database_returns_unknown(db):
    res = FaceRecognizer(db).identify(unit(0))
    assert res.decision == UNKNOWN and res.candidate_id is None
    assert "No identities" in res.reason


def test_highest_similarity_can_still_be_unknown(db):
    """The assignment example: Rahul 0.48, Anjali 0.41, Priya 0.35, threshold 0.60 -> UNKNOWN."""
    db.add_embeddings("rahul", "Rahul", [with_similarity(0, 1, 0.48)])
    db.add_embeddings("anjali", "Anjali", [with_similarity(0, 2, 0.41)])
    db.add_embeddings("priya", "Priya", [with_similarity(0, 3, 0.35)])
    res = FaceRecognizer(db, accept_threshold=0.60, uncertain_threshold=0.50).identify(unit(0))
    assert res.decision == UNKNOWN
    assert res.identity_label == UNKNOWN
    assert res.candidate_name == "Rahul"            # closest, but NOT accepted
    assert res.similarity == pytest.approx(0.48, abs=1e-6)
    assert [c[1] for c in res.top_candidates] == ["Rahul", "Anjali", "Priya"]


def test_recognized_and_borderline(db):
    db.add_embeddings("rahul", "Rahul", [with_similarity(0, 1, 0.84)])
    rec = FaceRecognizer(db, 0.70, 0.60)
    assert rec.identify(unit(0)).decision == RECOGNIZED
    assert rec.identify(unit(0)).identity_label == "Rahul"
    db.clear_database()
    db.add_embeddings("rahul", "Rahul", [with_similarity(0, 1, 0.65)])
    res = rec.identify(unit(0))
    assert res.decision == UNCERTAIN and res.identity_label == UNCERTAIN and res.candidate_name == "Rahul"


def test_identity_score_is_max_over_its_references(db):
    db.add_embeddings("p1", "Person One", [with_similarity(0, 1, 0.30), with_similarity(0, 2, 0.90)])
    db.add_embeddings("p2", "Person Two", [with_similarity(0, 3, 0.80)])
    res = FaceRecognizer(db, 0.70, 0.60).identify(unit(0))
    assert res.candidate_id == "p1" and res.similarity == pytest.approx(0.90, abs=1e-6)


def test_vectorised_ranking_equals_pairwise_cosine(db):
    rng = np.random.default_rng(1)
    refs = {f"id{i}": [rng.normal(size=DIM) for _ in range(3)] for i in range(5)}
    for pid, embs in refs.items():
        db.add_embeddings(pid, pid, embs)
    q = rng.normal(size=DIM)
    ranked = FaceRecognizer(db).rank_identities(q)
    for pid, _, sim in ranked:
        assert sim == pytest.approx(max(cosine_similarity(q, e) for e in refs[pid]), abs=1e-5)
    assert [r[2] for r in ranked] == sorted([r[2] for r in ranked], reverse=True)


def test_invalid_query_embedding_rejected(db):
    db.add_embeddings("p1", "P", [unit(0)])
    with pytest.raises(ValueError):
        FaceRecognizer(db).identify(np.zeros(DIM))
    with pytest.raises(ValueError):
        FaceRecognizer(db).identify(np.ones(10))


# ---------------------------------------------------------------- database
def test_database_persists_between_instances(tmp_path):
    db = FaceDatabase(db_dir=tmp_path)
    assert db.add_embeddings("p1", "John Doe", [unit(0)]) == 1
    assert db.add_embeddings("p1", "John Doe", [unit(1)]) == 2
    db2 = FaceDatabase(db_dir=tmp_path)
    assert db2.get_person("p1")["name"] == "John Doe"
    assert db2.num_embeddings == 2
    stored = np.array(db2.get_person("p1")["embeddings"][1])
    assert np.allclose(stored, unit(1))
    raw = json.loads((tmp_path / "database.json").read_text())
    assert raw["embedding_model"] == config.EMBEDDING_MODEL_NAME and raw["embedding_dim"] == DIM
    assert not list(tmp_path.glob("*.tmp")), "atomic save must not leave temp files"


def test_embeddings_are_normalised_on_storage(db):
    db.add_embeddings("p1", "P", [3.0 * unit(0)])
    assert np.linalg.norm(db.get_person("p1")["embeddings"][0]) == pytest.approx(1.0)


def test_same_id_different_name_rejected(db):
    db.add_embeddings("p1", "John Doe", [unit(0)])
    with pytest.raises(ValueError, match="already enrolled"):
        db.add_embeddings("p1", "Someone Else", [unit(1)])
    assert db.num_embeddings == 1


@pytest.mark.parametrize("bad", [np.ones(10), np.full(DIM, np.nan), np.zeros(DIM)])
def test_invalid_embeddings_rejected(db, bad):
    with pytest.raises(ValueError):
        db.add_embeddings("p1", "P", [bad])
    assert db.num_identities == 0


def test_corrupted_database_file_raises_and_is_not_overwritten(tmp_path):
    path = tmp_path / "database.json"
    path.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(DatabaseError):
        FaceDatabase(db_dir=tmp_path)
    assert path.read_text(encoding="utf-8") == "{ this is not json"


def test_invalid_embedding_data_in_file_raises(tmp_path):
    (tmp_path / "database.json").write_text(json.dumps(
        {"schema_version": 2, "embedding_model": config.EMBEDDING_MODEL_NAME, "embedding_dim": DIM,
         "identities": {"p1": {"name": "P", "embeddings": [[0.1, 0.2]]}}}), encoding="utf-8")
    with pytest.raises(DatabaseError, match="invalid embedding"):
        FaceDatabase(db_dir=tmp_path)


def test_model_mismatch_raises(tmp_path):
    (tmp_path / "database.json").write_text(json.dumps(
        {"schema_version": 2, "embedding_model": "some-other-model", "embedding_dim": DIM, "identities": {}}),
        encoding="utf-8")
    with pytest.raises(DatabaseError, match="different models"):
        FaceDatabase(db_dir=tmp_path)


def test_legacy_v1_file_loads(tmp_path):
    (tmp_path / "database.json").write_text(json.dumps(
        {"p1": {"name": "Old Format", "embeddings": [unit(0).tolist()]}}), encoding="utf-8")
    db = FaceDatabase(db_dir=tmp_path)
    assert db.get_person("p1")["name"] == "Old Format"


def test_missing_database_file_means_empty(tmp_path):
    db = FaceDatabase(db_dir=tmp_path / "does_not_exist_yet")
    assert db.num_identities == 0


def test_backup_corrupt_database_allows_fresh_start(tmp_path):
    (tmp_path / "database.json").write_text("garbage", encoding="utf-8")
    with pytest.raises(DatabaseError):
        FaceDatabase(db_dir=tmp_path)
    backup = backup_corrupt_database(tmp_path / "database.json")
    assert backup.read_text(encoding="utf-8") == "garbage"
    assert FaceDatabase(db_dir=tmp_path).num_identities == 0


def test_remove_and_clear(db):
    db.add_embeddings("p1", "A", [unit(0)])
    db.add_embeddings("p2", "B", [unit(1)])
    assert db.remove_person("p1") and not db.remove_person("p1")
    assert list(db.get_all_identities()) == ["p2"]
    db.clear_database()
    assert db.num_identities == 0 and db.get_embedding_matrix()[0].shape == (0, DIM)


@pytest.mark.parametrize("pid", ["", "  ", "has space", "-starts-with-dash", "x" * 65, "semi;colon"])
def test_invalid_person_ids(pid):
    with pytest.raises(ValueError):
        validate_person_id(pid)


def test_valid_ids_and_names():
    assert validate_person_id(" emp_001 ") == "emp_001"
    assert validate_name("  Anjali   Rao ") == "Anjali Rao"
    with pytest.raises(ValueError):
        validate_name("   ")


# ---------------------------------------------------------------- verification session
def result(decision_sim: float, accept=0.70, uncertain=0.60, pid="rahul"):
    from src.recognizer import MatchResult
    return MatchResult(decide(decision_sim, accept, uncertain), decision_sim, pid, pid.title(), accept, uncertain)


def test_session_recognized_on_first_attempt():
    s = VerificationSession(0.70, 0.60)
    assert s.record(result(0.84)) == RECOGNIZED
    assert s.is_complete and s.attempts_used == 1 and s.attempts_remaining == 0


def test_session_uncertain_then_recognized():
    s = VerificationSession(0.70, 0.60, max_attempts=3)
    assert s.record(result(0.65)) is None
    assert not s.is_complete and s.next_attempt_number == 2 and s.attempts_remaining == 2
    assert "attempt 2 of 3" in s.status_message()
    assert s.record(result(0.72)) == RECOGNIZED
    assert s.final_result.similarity == pytest.approx(0.72)


def test_session_uncertain_then_unknown():
    s = VerificationSession(0.70, 0.60)
    s.record(result(0.65))
    assert s.record(result(0.30)) == UNKNOWN


def test_session_three_uncertain_results_end_as_unable_to_verify():
    s = VerificationSession(0.70, 0.60, max_attempts=3)
    assert s.record(result(0.65)) is None
    assert s.record(result(0.64)) is None
    assert s.record(result(0.66)) == UNABLE_TO_VERIFY
    assert s.is_complete and s.attempts_used == 3
    assert "NOT recognized" in s.status_message()


def test_session_refuses_attempts_after_completion_no_infinite_loop():
    s = VerificationSession(0.70, 0.60, max_attempts=3)
    for sim in (0.65, 0.64, 0.66):
        s.record(result(sim))
    with pytest.raises(VerificationSessionClosed):
        s.record(result(0.99))
    assert s.attempts_used == 3 and s.final_decision == UNABLE_TO_VERIFY


def test_bounded_even_if_caller_loops_forever():
    s = VerificationSession(0.70, 0.60, max_attempts=3)
    submitted = 0
    while not s.is_complete:  # a buggy caller that keeps submitting borderline photos
        s.record(result(0.65))
        submitted += 1
        assert submitted <= 3
    assert submitted == 3


@pytest.mark.parametrize("max_attempts", [1, 2, 5])
def test_max_attempts_is_configurable(max_attempts):
    s = VerificationSession(0.70, 0.60, max_attempts=max_attempts)
    for _ in range(max_attempts - 1):
        assert s.record(result(0.65)) is None
    assert s.record(result(0.65)) == UNABLE_TO_VERIFY


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_invalid_max_attempts(bad):
    with pytest.raises(ValueError):
        VerificationSession(0.70, 0.60, max_attempts=bad)


def test_session_rejects_results_scored_with_other_thresholds():
    s = VerificationSession(0.70, 0.60)
    with pytest.raises(ValueError):
        s.record(result(0.65, accept=0.80, uncertain=0.50))


def test_new_session_starts_fresh():
    s1 = VerificationSession(0.70, 0.60)
    s1.record(result(0.65))
    s2 = VerificationSession(0.70, 0.60)
    assert s2.attempts_used == 0 and s2.session_id != s1.session_id and not s2.is_complete


# ---------------------------------------------------------------- image loading
def encode(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def test_corrupted_image_rejected():
    with pytest.raises(ImageLoadError):
        load_image(b"definitely not an image")
    good = encode(Image.new("RGB", (200, 200), "white"), "JPEG")
    with pytest.raises(ImageLoadError):
        load_image(good[: len(good) // 3])  # truncated file


@pytest.mark.parametrize("fmt", ["GIF", "TIFF"])
def test_unsupported_format_rejected(fmt):
    with pytest.raises(ImageLoadError, match="Unsupported"):
        load_image(encode(Image.new("RGB", (100, 100)), fmt))


def test_tiny_and_missing_images_rejected(tmp_path):
    with pytest.raises(ImageLoadError, match="too small"):
        load_image(encode(Image.new("RGB", (10, 10)), "PNG"))
    with pytest.raises(ImageLoadError, match="not found"):
        load_image(tmp_path / "missing.jpg")


def test_exif_orientation_applied():
    img = Image.new("RGB", (300, 200), "white")
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 CW to display" -- how phones store portrait photos
    out = load_image(encode(img, "JPEG", exif=exif.tobytes()))
    assert out.size == (200, 300)


def test_grayscale_rgba_converted_and_large_downscaled():
    assert load_image(encode(Image.new("L", (100, 100)), "PNG")).mode == "RGB"
    assert load_image(encode(Image.new("RGBA", (100, 100)), "PNG")).mode == "RGB"
    big = load_image(encode(Image.new("RGB", (4000, 2000)), "JPEG"))
    assert max(big.size) == config.MAX_IMAGE_SIDE and big.size[0] == 2 * big.size[1]


def test_crop_with_zero_padding_keeps_shape_at_border():
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    crop = _crop_with_zero_padding(img, 80, -10, 130, 40)
    assert crop.shape == (50, 50, 3)
    assert crop[:10].max() == 0 and crop[:, 20:].max() == 0  # outside the image -> zeros
    assert crop[10:, :20].min() == 200                        # inside the image -> original pixels


def test_draw_faces_handles_boxes_at_and_beyond_edges():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    out = draw_faces(img, [{"box": [-5, -5, 50, 60], "label": "UNKNOWN 0.12"},
                           {"box": [100, 0, 200, 130], "label": "Rahul 0.91", "color": (0, 255, 0)}])
    assert out.shape == img.shape and out.any()


# ---------------------------------------------------------------- pipeline orchestration (stub models)
def fake_face(key: int, size: float = 120.0, prob: float = 0.999, sharpness: float = 200.0,
              brightness: float = 120.0, x: float = 10.0) -> DetectedFace:
    """The face tensor carries `key`, which the stub embedder maps to a chosen embedding."""
    return DetectedFace([x, 10.0, x + size, 10.0 + size], prob, [[0, 0]] * 5,
                        torch.full((3, 160, 160), float(key)), FaceQuality(size, sharpness, brightness))


class StubDetector:
    def __init__(self, faces_by_width):
        self.faces_by_width = faces_by_width

    def detect_faces(self, image):
        return self.faces_by_width[image.width]


class StubEmbedder:
    def __init__(self, table):
        self.table = table

    def get_embeddings(self, tensors):
        out = np.array([self.table[int(t[0, 0, 0])] for t in tensors], dtype=np.float32).reshape(-1, DIM)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def make_system(db, faces_by_width, table):
    return FaceRecognitionSystem(db, StubDetector(faces_by_width), StubEmbedder(table))


def img(width: int) -> Image.Image:
    return Image.new("RGB", (width, 200))


def test_enroll_accepts_good_images_and_reports_bad_ones(db):
    faces = {
        101: [fake_face(1)],                          # good
        102: [fake_face(2)],                          # good, same person
        103: [],                                      # no face
        104: [fake_face(1), fake_face(3, x=150)],     # two faces
        105: [fake_face(4, size=30)],                 # too small
        106: [fake_face(5, sharpness=2.0)],           # blurry
        107: [fake_face(1)],                          # exact duplicate of image 101
        108: [fake_face(6)],                          # a different person
    }
    table = {1: with_similarity(0, 1, 0.9), 2: with_similarity(0, 2, 0.9), 3: unit(3), 4: unit(4),
             5: unit(5), 6: unit(6)}
    system = make_system(db, faces, table)
    report = system.enroll("rahul", "Rahul", [(f"img{w}", img(w)) for w in faces]
                           + [("corrupt", b"not an image")], accept_threshold=0.7, uncertain_threshold=0.6)
    by_label = {o.label: o for o in report.outcomes}
    assert by_label["img101"].accepted and by_label["img102"].accepted
    assert "No face" in by_label["img103"].message
    assert "2 faces" in by_label["img104"].message
    assert "too small" in by_label["img105"].message
    assert "blurry" in by_label["img106"].message
    assert "Duplicate" in by_label["img107"].message
    assert "does not match the other photos" in by_label["img108"].message
    assert not by_label["corrupt"].accepted
    assert report.stored_count == 2 and db.num_embeddings == 2
    assert report.warnings  # fewer than RECOMMENDED_ENROLL_IMAGES references


def test_enroll_rejects_existing_id_unless_adding(db):
    system = make_system(db, {101: [fake_face(1)], 102: [fake_face(2)]},
                         {1: with_similarity(0, 1, 0.9), 2: with_similarity(0, 2, 0.9)})
    system.enroll("rahul", "Rahul", [("a", img(101))])
    with pytest.raises(EnrollmentError, match="already enrolled"):
        system.enroll("rahul", "Rahul", [("b", img(102))])
    with pytest.raises(EnrollmentError, match="belongs to"):
        system.enroll("rahul", "Priya", [("b", img(102))], add_to_existing=True)
    report = system.enroll("rahul", "Rahul", [("b", img(102))], add_to_existing=True)
    assert report.success and report.total_references == 2


def test_enroll_blocks_same_face_under_a_second_id(db):
    system = make_system(db, {101: [fake_face(1)], 102: [fake_face(2)]},
                         {1: with_similarity(0, 1, 0.9), 2: with_similarity(0, 2, 0.9)})
    system.enroll("rahul", "Rahul", [("a", img(101))], accept_threshold=0.7, uncertain_threshold=0.6)
    report = system.enroll("rahul2", "Rahul Again", [("b", img(102))], accept_threshold=0.7, uncertain_threshold=0.6)
    assert not report.success and "already matches enrolled identity" in report.outcomes[0].message


@pytest.mark.parametrize("pid,name,images", [("bad id", "X", [("a", None)]), ("ok", "", [("a", None)]),
                                             ("ok", "Name", [])])
def test_enroll_invalid_requests(db, pid, name, images):
    with pytest.raises(EnrollmentError):
        make_system(db, {}, {}).enroll(pid, name, images)


def test_identify_multiple_faces_independently(db):
    db.add_embeddings("rahul", "Rahul", [unit(0)])
    faces = {300: [fake_face(2, x=200), fake_face(1, x=10), fake_face(3, x=100)]}
    table = {1: with_similarity(0, 5, 0.84), 2: with_similarity(0, 6, 0.43), 3: with_similarity(0, 7, 0.66)}
    results = make_system(db, faces, table).identify(img(300), accept_threshold=0.70, uncertain_threshold=0.60)
    assert [r.index for r in results] == [1, 2, 3]
    assert [r.match.decision for r in results] == [RECOGNIZED, UNCERTAIN, UNKNOWN]  # numbered left to right
    assert [round(r.match.similarity, 2) for r in results] == [0.84, 0.66, 0.43]


def test_verify_single_requires_exactly_one_face(db):
    db.add_embeddings("rahul", "Rahul", [unit(0)])
    system = make_system(db, {100: [], 200: [fake_face(1), fake_face(2, x=150)], 300: [fake_face(1)]},
                         {1: with_similarity(0, 5, 0.65), 2: unit(9)})
    with pytest.raises(NoFaceError):
        system.verify_single(img(100), 0.70, 0.60)
    with pytest.raises(MultipleFacesError):
        system.verify_single(img(200), 0.70, 0.60)
    assert system.verify_single(img(300), 0.70, 0.60).match.decision == UNCERTAIN
