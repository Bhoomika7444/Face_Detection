"""Critical false-acceptance test with group images.

    python scripts/group_false_acceptance_test.py --data data/lfw_eval/test

Each trial builds a COMPOSITE group image (4 LFW photos side by side, random order):
Person A (enrolled from *different* photos) + persons B, C, D who are never enrolled.
It then runs the exact identification code the app uses and checks, per face:
  * all 4 people are detected,
  * A is RECOGNIZED as A (or UNCERTAIN), never as someone else,
  * B/C/D are NOT recognized (UNKNOWN, or UNCERTAIN at worst).
Scenario "only_A": the database contains only A (the assignment's test).
Scenario "full_gallery": all ~150 test-split identities are enrolled (harder: more
chances for a stranger to resemble someone).

Nothing is hard-coded: A, B, C, D, their photos and positions are drawn at random (seeded).
These are composites of separate photos, not real group photographs; for a real group photo,
enroll the person in the app and upload the photo in the Identify tab.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.database import FaceDatabase  # noqa: E402
from src.evaluation import load_protocol  # noqa: E402
from src.pipeline import FaceRecognitionSystem  # noqa: E402
from src.recognizer import RECOGNIZED, UNCERTAIN, UNKNOWN  # noqa: E402
from src.utils import cv2_to_pil, draw_faces, load_image, pil_to_cv2  # noqa: E402

TILE = 250


def compose(paths: list[Path]) -> Image.Image:
    canvas = Image.new("RGB", (TILE * len(paths), TILE))
    for i, p in enumerate(paths):
        canvas.paste(load_image(p).resize((TILE, TILE)), (i * TILE, 0))
    return canvas


def assign_to_tiles(results, n_tiles: int):
    """Map each detected face to the tile containing its centre; the face nearest the tile
    centre is that tile's labelled person, any others are background people in that photo."""
    tiles = {i: [] for i in range(n_tiles)}
    for r in results:
        cx, cy = r.face.center
        tiles[max(0, min(int(cx // TILE), n_tiles - 1))].append(r)  # faces may extend past the image edge
    subjects, background = {}, []
    for i, faces in tiles.items():
        if not faces:
            continue
        centre = (i * TILE + TILE / 2, TILE / 2)
        faces.sort(key=lambda r: (r.face.center[0] - centre[0]) ** 2 + (r.face.center[1] - centre[1]) ** 2)
        subjects[i] = faces[0]
        background.extend(faces[1:])
    return subjects, background


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "lfw_eval" / "test")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--accept", type=float, default=config.ACCEPT_THRESHOLD)
    ap.add_argument("--uncertain", type=float, default=config.UNCERTAIN_THRESHOLD)
    ap.add_argument("--out", type=Path, default=config.RESULTS_DIR / "group_test")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    protocol = load_protocol(args.data)
    known_by_id: dict[str, list[Path]] = {}
    for pid, p in protocol.known_probes:
        known_by_id.setdefault(pid, []).append(p)
    unknown_by_id: dict[str, list[Path]] = {}
    for uid, p in protocol.unknown_probes:
        unknown_by_id.setdefault(uid, []).append(p)
    args.out.mkdir(parents=True, exist_ok=True)
    examples_dir = args.out / "examples"
    examples_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        full_db = FaceDatabase(db_dir=Path(tmp) / "full")
        system = FaceRecognitionSystem(full_db)
        print("Enrolling the full test gallery with the app's enrollment code ...")
        enrolled_full = set()
        for pid, paths in protocol.gallery.items():
            rep = system.enroll(pid, pid.replace("_", " "), [(p.name, p) for p in paths],
                                accept_threshold=args.accept, uncertain_threshold=args.uncertain)
            if rep.success:
                enrolled_full.add(pid)
        print(f"  {len(enrolled_full)} of {len(protocol.gallery)} identities enrolled")

        candidates_a = sorted(pid for pid in enrolled_full if pid in known_by_id)
        unknown_ids = sorted(unknown_by_id)
        scenarios = {"only_A": Counter(), "full_gallery": Counter()}
        trials_log = []
        for t in range(args.trials):
            a = rng.choice(candidates_a)
            strangers = rng.sample(unknown_ids, 3)
            people = [(a, rng.choice(known_by_id[a]), True)] + [(u, rng.choice(unknown_by_id[u]), False) for u in strangers]
            rng.shuffle(people)  # random position for A
            composite = compose([p for _, p, _ in people])

            only_a_db = FaceDatabase(db_dir=Path(tmp) / f"only_a_{t}")
            only_a_db.add_embeddings(a, a.replace("_", " "), [e for e in full_db.get_person(a)["embeddings"]])
            for name, db in (("only_A", only_a_db), ("full_gallery", full_db)):
                c = scenarios[name]
                results = FaceRecognitionSystem(db, system.detector, system.embedder).identify(
                    composite, args.accept, args.uncertain)
                subjects, background = assign_to_tiles(results, len(people))
                c["trials"] += 1
                c["faces_detected_total"] += len(results)
                c["all_4_people_detected"] += int(len(subjects) == 4)
                row = {"trial": t, "scenario": name, "faces_detected": len(results), "tiles": []}
                for i, (pid, path, is_a) in enumerate(people):
                    r = subjects.get(i)
                    role = "A" if is_a else "stranger"
                    if r is None:
                        c[f"{role}_not_detected"] += 1
                        row["tiles"].append({"role": role, "detected": False})
                        continue
                    m = r.match
                    if is_a:
                        outcome = ("A_recognized_correctly" if m.decision == RECOGNIZED and m.candidate_id == a
                                   else "A_recognized_as_wrong_person" if m.decision == RECOGNIZED
                                   else f"A_{m.decision.lower()}")
                    else:
                        outcome = "stranger_FALSELY_ACCEPTED" if m.decision == RECOGNIZED else f"stranger_{m.decision.lower()}"
                    c[outcome] += 1
                    row["tiles"].append({"role": role, "decision": m.decision, "candidate": m.candidate_id,
                                         "similarity": round(m.similarity, 4)})
                for r in background:
                    c[f"background_face_{r.match.decision.lower()}"] += 1
                trials_log.append(row)
                if t < 3:
                    colors = {RECOGNIZED: (0, 170, 0), UNCERTAIN: (0, 165, 255), UNKNOWN: (0, 0, 220)}
                    drawn = draw_faces(pil_to_cv2(composite), [
                        {"box": r.face.box, "color": colors[r.match.decision],
                         "label": f"{r.match.identity_label[:14]} {r.match.similarity:.2f}"} for r in results])
                    cv2_to_pil(drawn).save(examples_dir / f"trial{t}_{name}.png")

    summary = {"description": "Composite group images: 1 enrolled person (A) + 3 never-enrolled people",
               "data": args.data.name, "trials": args.trials, "seed": args.seed,
               "accept_threshold": args.accept, "uncertain_threshold": args.uncertain,
               "scenarios": {k: dict(sorted(v.items())) for k, v in scenarios.items()}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "trials.json").write_text(json.dumps(trials_log, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
