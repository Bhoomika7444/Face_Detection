"""Reproducible open-set identification evaluation.

Dataset layout (works for LFW splits from scripts/prepare_lfw.py and for your own photos):

    <data>/enrolled/<person_id>/*.jpg       photos used to enroll each person (the gallery)
    <data>/test/known/<person_id>/*.jpg     OTHER photos of enrolled people  -> should be RECOGNIZED
    <data>/test/unknown/<any_id>/*.jpg      people who are NOT enrolled      -> should be UNKNOWN

Usage:
    python -m src.evaluation --data data/lfw_eval/val --calibrate --out results/lfw_val
    python -m src.evaluation --data data/lfw_eval/test --thresholds-from results/lfw_val/calibration.json --out results/lfw_test

Every image runs through the same detection -> alignment -> embedding code as the app. Models
run once per image (results are cached); thresholds are then swept on the stored scores.
When a photo contains several faces, the face nearest the image centre is taken as the
labelled person (LFW photos are centred on the labelled person).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config
from .database import FaceDatabase
from .recognizer import RECOGNIZED, UNCERTAIN, UNKNOWN, FaceRecognizer, MatchResult, decide, validate_thresholds
from .verification import UNABLE_TO_VERIFY, VerificationSession

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Bump when preprocessing changes so cached embeddings are recomputed.
PIPELINE_TAG = (f"{config.EMBEDDING_MODEL_NAME}|det{config.MIN_DETECTION_PROB}|max{config.MAX_IMAGE_SIDE}|"
                f"mtcnn{config.MTCNN_MIN_FACE_SIZE}{config.MTCNN_STAGE_THRESHOLDS}|align-v2-zeropad|central-face")
THRESHOLD_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)


# ======================================================================================
# Dataset
# ======================================================================================
@dataclass
class Protocol:
    root: Path
    gallery: dict[str, list[Path]]
    known_probes: list[tuple[str, Path]]
    unknown_probes: list[tuple[str, Path]]


def _images_by_identity(folder: Path) -> dict[str, list[Path]]:
    if not folder.is_dir():
        return {}
    out = {}
    for d in sorted(p for p in folder.iterdir() if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        if files:
            out[d.name] = files
    return out


def load_protocol(root: Path) -> Protocol:
    root = Path(root)
    gallery = _images_by_identity(root / "enrolled")
    known = _images_by_identity(root / "test" / "known")
    unknown = _images_by_identity(root / "test" / "unknown")
    if not gallery:
        raise SystemExit(f"No enrollment images found in {root / 'enrolled'}/<person_id>/")
    if not known and not unknown:
        raise SystemExit(f"No test images found in {root / 'test'}/known or /unknown")
    missing = sorted(set(known) - set(gallery))
    if missing:
        raise SystemExit(f"test/known identities without enrollment photos: {missing[:10]}")
    leaked = sorted(set(unknown) & set(gallery))
    if leaked:
        raise SystemExit(f"test/unknown identities must not be enrolled, but these are: {leaked[:10]}")
    return Protocol(root, gallery,
                    [(pid, p) for pid, ps in known.items() for p in ps],
                    [(uid, p) for uid, ps in unknown.items() for p in ps])


# ======================================================================================
# Running the models once per image (with a cache)
# ======================================================================================
@dataclass
class ImageRecord:
    status: str               # "ok" | "no_face" | "load_error"
    n_faces: int = 0
    prob: float = 0.0
    face_size: float = 0.0
    sharpness: float = 0.0
    brightness: float = 0.0
    error: str = ""
    embedding: np.ndarray | None = None


def embed_images(system, paths: list[Path], cache_dir: Path | None) -> dict[Path, ImageRecord]:
    from .detector import select_central_face
    from .utils import ImageLoadError, load_image

    records: dict[Path, ImageRecord] = {}
    cache: dict[str, dict] = {}
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"embeddings-{hashlib.sha1(PIPELINE_TAG.encode()).hexdigest()[:10]}.json"
        if cache_file.exists():
            cache = json.loads(cache_file.read_text(encoding="utf-8"))

    def key(p: Path) -> str:
        st = p.stat()
        return f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}"

    todo = []
    for p in paths:
        c = cache.get(key(p))
        if c is not None:
            emb = None if c.get("embedding") is None else np.asarray(c["embedding"], dtype=np.float32)
            records[p] = ImageRecord(**{k: v for k, v in c.items() if k != "embedding"}, embedding=emb)
        else:
            todo.append(p)

    t0 = time.time()
    for i, p in enumerate(todo, 1):
        try:
            image = load_image(p)
        except ImageLoadError as exc:
            records[p] = ImageRecord("load_error", error=str(exc))
            continue
        faces = system.detector.detect_faces(image)
        face = select_central_face(faces, image.size)
        if face is None:
            records[p] = ImageRecord("no_face")
            continue
        emb = system.embedder.get_embedding(face.face_tensor)
        records[p] = ImageRecord("ok", len(faces), face.prob, face.quality.face_size,
                                 face.quality.sharpness, face.quality.brightness, embedding=emb)
        if i % 250 == 0 or i == len(todo):
            print(f"  embedded {i}/{len(todo)} new images ({time.time() - t0:.0f}s)", flush=True)

    if cache_file is not None and todo:
        for p in todo:
            r = records[p]
            d = {k: v for k, v in asdict(r).items() if k != "embedding"}
            d["embedding"] = None if r.embedding is None else [round(float(x), 7) for x in r.embedding]
            cache[key(p)] = d
        cache_file.write_text(json.dumps(cache), encoding="utf-8")
    return records


# ======================================================================================
# Gallery + scoring
# ======================================================================================
@dataclass
class ProbeScore:
    role: str                  # "known" or "unknown"
    true_id: str
    path: str
    best_id: str
    best_sim: float
    genuine_sim: float | None  # score of the correct identity (known probes only)
    second_id: str | None
    second_sim: float | None

    @property
    def best_is_correct(self) -> bool:
        return self.role == "known" and self.best_id == self.true_id


def build_gallery(protocol: Protocol, records: dict[Path, ImageRecord], db_dir: Path):
    """Enroll every gallery identity with the same quality gates the app uses."""
    from .pipeline import quality_issues

    db = FaceDatabase(db_dir=db_dir)
    report = {"images": 0, "stored": 0, "no_face": 0, "rejected_quality": 0, "identities_without_references": []}
    for pid, paths in protocol.gallery.items():
        embs = []
        for p in paths:
            report["images"] += 1
            r = records[p]
            if r.status != "ok":
                report["no_face"] += 1
            elif quality_issues(r.face_size, r.prob, r.sharpness, r.brightness):
                report["rejected_quality"] += 1
            else:
                embs.append(r.embedding)
        if embs:
            db.add_embeddings(pid, pid.replace("_", " "), embs)
            report["stored"] += len(embs)
        else:
            report["identities_without_references"].append(pid)
    return db, report


def score_probes(protocol: Protocol, records: dict[Path, ImageRecord], db: FaceDatabase):
    recognizer = FaceRecognizer(db, accept_threshold=1.0, uncertain_threshold=1.0)  # ranking only
    scores: list[ProbeScore] = []
    failures = {"known_no_face": [], "unknown_no_face": [], "known_unenrollable": []}
    enrolled = set(db.get_all_identities())
    for role, probes in (("known", protocol.known_probes), ("unknown", protocol.unknown_probes)):
        for true_id, p in probes:
            rel = p.relative_to(protocol.root).as_posix()
            if role == "known" and true_id not in enrolled:
                failures["known_unenrollable"].append(rel)
                continue
            r = records[p]
            if r.status != "ok":
                failures[f"{role}_no_face"].append(rel)
                continue
            ranked = recognizer.rank_identities(r.embedding)
            genuine = next((s for pid, _, s in ranked if pid == true_id), None) if role == "known" else None
            second = ranked[1] if len(ranked) > 1 else (None, None, None)
            scores.append(ProbeScore(role, true_id, rel, ranked[0][0], ranked[0][2], genuine, second[0], second[2]))
    return scores, failures


# ======================================================================================
# Metrics
# ======================================================================================
def _arrays(scores: list[ProbeScore]):
    known = [s for s in scores if s.role == "known"]
    unknown = [s for s in scores if s.role == "unknown"]
    kb = np.array([s.best_sim for s in known])
    kc = np.array([s.best_is_correct for s in known], dtype=bool)
    ub = np.array([s.best_sim for s in unknown])
    return kb, kc, ub


def operating_point_metrics(scores: list[ProbeScore], accept: float, uncertain: float) -> dict:
    """Three-state (RECOGNIZED / UNCERTAIN / UNKNOWN) rates for one attempt."""
    kb, kc, ub = _arrays(scores)
    k_dec = np.array([decide(s, accept, uncertain) for s in kb])
    u_dec = np.array([decide(s, accept, uncertain) for s in ub])
    nk, nu = max(len(kb), 1), max(len(ub), 1)
    known = {
        "count": int(len(kb)),
        "correctly_recognized": float(np.sum((k_dec == RECOGNIZED) & kc) / nk),
        "misidentified_as_other_person": float(np.sum((k_dec == RECOGNIZED) & ~kc) / nk),
        "uncertain": float(np.sum(k_dec == UNCERTAIN) / nk),
        "rejected_as_unknown": float(np.sum(k_dec == UNKNOWN) / nk),
    }
    unknown = {
        "count": int(len(ub)),
        "correctly_rejected_unknown": float(np.sum(u_dec == UNKNOWN) / nu),
        "uncertain": float(np.sum(u_dec == UNCERTAIN) / nu),
        "falsely_accepted": float(np.sum(u_dec == RECOGNIZED) / nu),
    }
    # Two-state view of the same operating point (UNCERTAIN counted as "not accepted").
    two_state = {
        "FAR_unknown_accepted": unknown["falsely_accepted"],
        "FRR_known_not_correctly_accepted": 1.0 - known["correctly_recognized"],
        "misidentification_rate": known["misidentified_as_other_person"],
    }
    return {"accept_threshold": accept, "uncertain_threshold": uncertain,
            "known": known, "unknown": unknown, "two_state": two_state}


def threshold_sweep(scores: list[ProbeScore], grid=THRESHOLD_GRID) -> list[dict]:
    """Single-threshold (no UNCERTAIN band) FAR / FRR / mis-ID at every threshold."""
    kb, kc, ub = _arrays(scores)
    rows = []
    for t in grid:
        acc_k = kb >= t
        rows.append({
            "threshold": float(t),
            "FAR": float(np.mean(ub >= t)) if len(ub) else 0.0,
            "FRR": float(1 - np.mean(acc_k & kc)) if len(kb) else 0.0,
            "misID": float(np.mean(acc_k & ~kc)) if len(kb) else 0.0,
            "known_below": float(np.mean(kb < t)) if len(kb) else 0.0,
        })
    return rows


def equal_error_rate(sweep: list[dict]) -> dict:
    best = min(sweep, key=lambda r: abs(r["FAR"] - r["FRR"]))
    return {"threshold": best["threshold"], "FAR": best["FAR"], "FRR": best["FRR"]}


def select_thresholds(scores: list[ProbeScore], target_far: float, target_known_reject: float,
                      max_attempts: int) -> dict:
    """Calibration rule (documented in the README):
    UNCERTAIN(a) = highest threshold <= a at which at most target_known_reject of enrolled people
                   are rejected outright as UNKNOWN (controls outright false rejection).
    ACCEPT       = lowest threshold a such that, with the band [UNCERTAIN(a), a):
                   - single-attempt FAR and mis-ID rate are <= target_far, and
                   - FAR and mis-ID after the full re-verification workflow (up to max_attempts
                     photos) are also <= target_far.
    The second condition matters: retries give an unknown person in the band extra chances."""
    sweep = threshold_sweep(scores)
    rejected = []
    for r in sweep:
        if r["FAR"] > target_far or r["misID"] > target_far:
            continue
        accept = r["threshold"]
        band_ok = [b for b in sweep if b["threshold"] <= accept and b["known_below"] <= target_known_reject]
        uncertain = band_ok[-1]["threshold"] if band_ok else accept
        sessions = simulate_sessions(scores, accept, uncertain, max_attempts)
        session_far = sessions["unknown"]["after_reverification"].get("falsely_accepted", 0.0)
        session_misid = sessions["known"]["after_reverification"].get("misidentified", 0.0)
        if session_far <= target_far and session_misid <= target_far:
            return {"accept_threshold": accept, "uncertain_threshold": uncertain,
                    "rule": {"target_far": target_far, "target_known_reject": target_known_reject,
                             "max_attempts": max_attempts},
                    "at_accept": r,
                    "at_uncertain": next(b for b in sweep if b["threshold"] == uncertain),
                    "session_far": session_far, "session_misid": session_misid,
                    "rejected_candidates": rejected}
        rejected.append({"accept": accept, "uncertain": uncertain, "session_far": round(session_far, 4),
                         "reason": "FAR after re-verification above target"})
    raise SystemExit("No threshold pair meets the FAR target on this data.")


def simulate_sessions(scores: list[ProbeScore], accept: float, uncertain: float, max_attempts: int) -> dict:
    """Replay the re-verification workflow with the real VerificationSession class.

    For every identity with at least max_attempts test photos, each rotation of its photo list
    is one session: attempt 1 uses the first photo, and a further photo is used only while
    the result stays UNCERTAIN. The first-attempt outcome of the same sessions is reported
    alongside, to show exactly what re-verification changes (including extra FAR)."""
    groups: dict[tuple[str, str], list[ProbeScore]] = defaultdict(list)
    for s in scores:
        groups[(s.role, s.true_id)].append(s)

    def label(role, s_final, result: MatchResult | None, true_id):
        if s_final == RECOGNIZED:
            if role == "unknown":
                return "falsely_accepted"
            return "correctly_recognized" if result.candidate_id == true_id else "misidentified"
        return {UNKNOWN: "rejected_as_unknown", UNABLE_TO_VERIFY: "unable_to_verify"}[s_final]

    out = {}
    for role in ("known", "unknown"):
        final, first, attempts_hist = Counter(), Counter(), Counter()
        n = 0
        for (r, true_id), probes in groups.items():
            if r != role or len(probes) < max_attempts:
                continue
            for start in range(len(probes)):
                seq = probes[start:] + probes[:start]
                session = VerificationSession(accept, uncertain, max_attempts)
                for p in seq:
                    res = MatchResult(decide(p.best_sim, accept, uncertain), p.best_sim, p.best_id, p.best_id,
                                      accept, uncertain)
                    if session.attempts_used == 0:
                        d = res.decision
                        first[label(role, d, res, true_id) if d != UNCERTAIN else "uncertain"] += 1
                    if session.record(res):
                        break
                n += 1
                attempts_hist[session.attempts_used] += 1
                final[label(role, session.final_decision, session.final_result, true_id)] += 1
        out[role] = {"sessions": n,
                     "after_reverification": {k: v / n for k, v in sorted(final.items())} if n else {},
                     "first_attempt_only": {k: v / n for k, v in sorted(first.items())} if n else {},
                     "attempts_used_histogram": dict(sorted(attempts_hist.items()))}
    return out


def decision_confusion(scores: list[ProbeScore], accept: float, uncertain: float) -> tuple[list[str], dict]:
    """Rows: true identity (or UNKNOWN_PERSON). Columns: predicted identity / UNCERTAIN / UNKNOWN."""
    matrix: dict[str, Counter] = defaultdict(Counter)
    for s in scores:
        d = decide(s.best_sim, accept, uncertain)
        pred = s.best_id if d == RECOGNIZED else d
        matrix["UNKNOWN_PERSON" if s.role == "unknown" else s.true_id][pred] += 1
    cols = sorted({c for row in matrix.values() for c in row} - {UNCERTAIN, UNKNOWN}) + [UNCERTAIN, UNKNOWN]
    return cols, matrix


# ======================================================================================
# Outputs
# ======================================================================================
def save_plots(scores: list[ProbeScore], sweep: list[dict], accept: float, uncertain: float, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kb, kc, ub = _arrays(scores)
    bins = np.linspace(-0.3, 1.0, 66)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    # Each group is normalised to sum to 1, so groups of different sizes are comparable.
    ax.hist(ub, bins=bins, alpha=0.6, weights=np.full(len(ub), 1 / max(len(ub), 1)), color="#d62728",
            label=f"unknown people: best match (n={len(ub)})")
    ax.hist(kb[kc], bins=bins, alpha=0.6, weights=np.full(int(kc.sum()), 1 / max(int(kc.sum()), 1)),
            color="#2ca02c", label=f"enrolled people: best match is correct (n={int(kc.sum())})")
    if (~kc).any():  # usually a handful of photos: show each one as a marker instead of a histogram
        ax.plot(kb[~kc], np.zeros(int((~kc).sum())), "x", color="#ff7f0e", ms=10, mew=2, clip_on=False,
                label=f"enrolled people: best match is WRONG person (n={int((~kc).sum())})")
    ax.axvspan(uncertain, accept, color="#bcbd22", alpha=0.18, label="UNCERTAIN band")
    ax.axvline(accept, color="k", lw=1.5, label=f"accept = {accept:.2f}")
    ax.axvline(uncertain, color="k", lw=1, ls="--", label=f"uncertain = {uncertain:.2f}")
    ax.set_xlabel("cosine similarity of best-matching enrolled identity")
    ax.set_ylabel("fraction of photos in group")
    ax.set_title("Similarity score distributions (open-set identification)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "score_distributions.png", dpi=120)
    plt.close(fig)

    t = [r["threshold"] for r in sweep]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t, [r["FAR"] for r in sweep], color="#d62728", label="FAR: unknown person accepted")
    ax.plot(t, [r["FRR"] for r in sweep], color="#1f77b4", label="FRR: enrolled person not correctly accepted")
    ax.plot(t, [r["misID"] for r in sweep], color="#ff7f0e", label="mis-ID: accepted as the wrong enrolled person")
    ax.axvline(accept, color="k", lw=1.5)
    ax.axvline(uncertain, color="k", lw=1, ls="--")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("acceptance threshold (single threshold, no UNCERTAIN band)")
    ax.set_ylabel("rate (symlog scale)")
    ax.set_title("Error rates vs. threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "error_rates_vs_threshold.png", dpi=120)
    plt.close(fig)
    return ["score_distributions.png", "error_rates_vs_threshold.png"]


def failure_cases(scores: list[ProbeScore], accept: float, uncertain: float, k: int = 10) -> dict:
    def row(s: ProbeScore):
        return {"image": s.path, "true_id": s.true_id if s.role == "known" else "(not enrolled)",
                "decision": decide(s.best_sim, accept, uncertain), "best_match": s.best_id,
                "best_similarity": round(s.best_sim, 4),
                "correct_identity_similarity": None if s.genuine_sim is None else round(s.genuine_sim, 4)}
    known = [s for s in scores if s.role == "known"]
    unknown = [s for s in scores if s.role == "unknown"]
    return {
        "enrolled_people_lowest_scores": [row(s) for s in sorted(known, key=lambda s: s.best_sim)[:k]],
        "enrolled_people_misidentified": [row(s) for s in known if not s.best_is_correct and s.best_sim >= uncertain][:k],
        "unknown_people_highest_scores": [row(s) for s in sorted(unknown, key=lambda s: -s.best_sim)[:k]],
    }


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def write_report(summary: dict, out: Path) -> None:
    m = summary["operating_point"]
    lines = [
        f"# Evaluation report: `{summary['dataset']}`", "",
        f"Generated by `python -m src.evaluation` on {summary['generated_at']}. All numbers are measured, not estimated.", "",
        f"* Enrolled identities: **{summary['gallery']['identities_enrolled']}** "
        f"({summary['gallery']['stored']} reference embeddings)",
        f"* Known-person test photos: **{m['known']['count']}**, unknown-person test photos: **{m['unknown']['count']}**",
        f"* Thresholds: accept = **{m['accept_threshold']:.2f}**, uncertain = **{m['uncertain_threshold']:.2f}** "
        f"({summary['threshold_source']})", "",
        "## Single attempt (three-state decision)", "",
        "| Enrolled people (should be RECOGNIZED) | rate |", "|---|---|",
        f"| correctly recognized | {pct(m['known']['correctly_recognized'])} |",
        f"| misidentified as another enrolled person | {pct(m['known']['misidentified_as_other_person'])} |",
        f"| UNCERTAIN (asked for another photo) | {pct(m['known']['uncertain'])} |",
        f"| rejected as UNKNOWN (false rejection) | {pct(m['known']['rejected_as_unknown'])} |", "",
        "| Unknown people (should be UNKNOWN) | rate |", "|---|---|",
        f"| correctly rejected as UNKNOWN | {pct(m['unknown']['correctly_rejected_unknown'])} |",
        f"| UNCERTAIN (asked for another photo) | {pct(m['unknown']['uncertain'])} |",
        f"| falsely accepted (false acceptance) | {pct(m['unknown']['falsely_accepted'])} |", "",
        f"Two-state view at the accept threshold: FAR = {pct(m['two_state']['FAR_unknown_accepted'])}, "
        f"FRR = {pct(m['two_state']['FRR_known_not_correctly_accepted'])}. "
        f"Equal error rate: {pct(summary['equal_error_rate']['FAR'])} FAR / "
        f"{pct(summary['equal_error_rate']['FRR'])} FRR at threshold {summary['equal_error_rate']['threshold']:.2f}.", "",
        f"## Re-verification sessions (max {summary['max_attempts']} attempts)", "",
    ]
    for role in ("known", "unknown"):
        s = summary["sessions"][role]
        lines.append(f"**{role} people**: {s['sessions']} sessions")
        lines.append("")
        lines.append("| outcome | first attempt only | after re-verification |")
        lines.append("|---|---|---|")
        for k in sorted(set(s["first_attempt_only"]) | set(s["after_reverification"])):
            lines.append(f"| {k} | {pct(s['first_attempt_only'].get(k, 0))} | {pct(s['after_reverification'].get(k, 0))} |")
        lines.append("")
    f = summary["detection_failures"]
    lines += ["## Detection", "",
              f"* Known-person photos with no detectable face: {len(f['known_no_face'])}",
              f"* Unknown-person photos with no detectable face: {len(f['unknown_no_face'])}",
              f"* Gallery photos rejected by enrollment quality gates: {summary['gallery']['rejected_quality']}, "
              f"no face: {summary['gallery']['no_face']}", "",
              "Plots: `score_distributions.png`, `error_rates_vs_threshold.png`. Full numbers: `summary.json`, "
              "`threshold_sweep.csv`, `confusion_matrix.csv`."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(data: Path, out: Path, accept: float, uncertain: float, max_attempts: int, calibrate: bool,
        target_far: float, target_known_reject: float, threshold_source: str, device: str,
        use_cache: bool) -> dict:
    from .pipeline import FaceRecognitionSystem

    protocol = load_protocol(data)
    out.mkdir(parents=True, exist_ok=True)
    all_paths = ([p for ps in protocol.gallery.values() for p in ps]
                 + [p for _, p in protocol.known_probes] + [p for _, p in protocol.unknown_probes])
    print(f"Dataset {data}: {len(protocol.gallery)} enrolled identities, {len(protocol.known_probes)} known probes, "
          f"{len(protocol.unknown_probes)} unknown probes")

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        system = FaceRecognitionSystem(database=FaceDatabase(db_dir=tmp, db_name="unused.json"), device=device)
        records = embed_images(system, all_paths, config.RESULTS_DIR / "cache" if use_cache else None)
        db, gallery_report = build_gallery(protocol, records, Path(tmp))
        scores, failures = score_probes(protocol, records, db)
    elapsed = time.time() - t0

    calibration = None
    if calibrate:
        calibration = select_thresholds(scores, target_far, target_known_reject, max_attempts)
        accept, uncertain = calibration["accept_threshold"], calibration["uncertain_threshold"]
        threshold_source = f"calibrated on this dataset: {calibration['rule']}"
        (out / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    validate_thresholds(accept, uncertain)

    sweep = threshold_sweep(scores)
    with open(out / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)
    cols, matrix = decision_confusion(scores, accept, uncertain)
    with open(out / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + cols)
        for row in sorted(matrix, key=lambda r: (r == "UNKNOWN_PERSON", r)):
            w.writerow([row] + [matrix[row].get(c, 0) for c in cols])

    kb, kc, ub = _arrays(scores)
    summary = {
        "dataset": data.as_posix() if not data.is_absolute() else data.name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "pipeline": PIPELINE_TAG,
        "threshold_source": threshold_source,
        "max_attempts": max_attempts,
        "gallery": {**gallery_report, "identities_enrolled": db.num_identities,
                    "identities_without_references": len(gallery_report["identities_without_references"])},
        "operating_point": operating_point_metrics(scores, accept, uncertain),
        "equal_error_rate": equal_error_rate(sweep),
        "sessions": simulate_sessions(scores, accept, uncertain, max_attempts),
        "score_statistics": {
            "known_best_correct": {"mean": float(kb[kc].mean()), "p05": float(np.percentile(kb[kc], 5))} if kc.any() else {},
            "unknown_best": {"mean": float(ub.mean()), "p95": float(np.percentile(ub, 95)),
                             "p99": float(np.percentile(ub, 99)), "max": float(ub.max())} if len(ub) else {},
        },
        "detection_failures": {k: v for k, v in failures.items()},
        "failure_cases": failure_cases(scores, accept, uncertain),
        "calibration": calibration,
        "runtime_seconds": round(elapsed, 1),
        "plots": save_plots(scores, sweep, accept, uncertain, out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, out)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the face recognition system (open-set identification).")
    ap.add_argument("--data", type=Path, required=True, help="folder with enrolled/, test/known/, test/unknown/")
    ap.add_argument("--out", type=Path, default=config.RESULTS_DIR / "eval")
    ap.add_argument("--accept", type=float, default=config.ACCEPT_THRESHOLD)
    ap.add_argument("--uncertain", type=float, default=config.UNCERTAIN_THRESHOLD)
    ap.add_argument("--thresholds-from", type=Path, help="calibration.json produced by --calibrate")
    ap.add_argument("--max-attempts", type=int, default=config.MAX_VERIFICATION_ATTEMPTS)
    ap.add_argument("--calibrate", action="store_true", help="choose thresholds on THIS data (use a validation split)")
    ap.add_argument("--target-far", type=float, default=0.01)
    ap.add_argument("--target-known-reject", type=float, default=0.02)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    accept, uncertain, source = args.accept, args.uncertain, "command line / src/config.py"
    if args.thresholds_from:
        cal = json.loads(args.thresholds_from.read_text(encoding="utf-8"))
        accept, uncertain = cal["accept_threshold"], cal["uncertain_threshold"]
        source = f"from {args.thresholds_from.as_posix()} (chosen on a different, identity-disjoint split)"

    summary = run(args.data, args.out, accept, uncertain, args.max_attempts, args.calibrate, args.target_far,
                  args.target_known_reject, source, args.device, not args.no_cache)
    print((args.out / "report.md").read_text(encoding="utf-8"))
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
