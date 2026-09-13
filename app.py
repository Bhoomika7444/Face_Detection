"""Streamlit UI for the Face Recognition Identification System.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Face Recognition System", page_icon="🧑", layout="wide")

from src import config  # noqa: E402
from src.database import DatabaseError, FaceDatabase, backup_corrupt_database  # noqa: E402
from src.detector import FaceDetectionError, FaceDetector, ModelInitError  # noqa: E402
from src.embedder import EmbeddingError, FaceEmbedder  # noqa: E402
from src.pipeline import EnrollmentError, FaceRecognitionSystem, MultipleFacesError, NoFaceError  # noqa: E402
from src.recognizer import RECOGNIZED, UNCERTAIN, UNKNOWN  # noqa: E402
from src.utils import ImageLoadError, cv2_to_pil, draw_faces, load_image, pil_to_cv2  # noqa: E402
from src.verification import VerificationSession  # noqa: E402

BOX_COLORS_BGR = {RECOGNIZED: (0, 170, 0), UNCERTAIN: (0, 165, 255), UNKNOWN: (0, 0, 220)}
PIPELINE_ERRORS = (ImageLoadError, FaceDetectionError, EmbeddingError)


@st.cache_resource(show_spinner="Loading the face detection and recognition models ...")
def load_models():
    return FaceDetector(device="cpu"), FaceEmbedder(device="cpu")


@st.cache_resource
def load_database() -> FaceDatabase:
    return FaceDatabase()  # <project>/embeddings/database.json


# ----------------------------------------------------------------------------- sidebar
def sidebar_settings(db: FaceDatabase) -> tuple[float, float, int]:
    st.sidebar.header("Decision settings")
    if st.sidebar.button("Reset to calibrated defaults"):
        for k in ("accept", "uncertain", "max_attempts"):
            st.session_state.pop(k, None)
    accept = st.sidebar.slider("Recognition (acceptance) threshold", 0.30, 0.95, config.ACCEPT_THRESHOLD, 0.01,
                               key="accept", help="Similarity >= this value -> RECOGNIZED.")
    uncertain = st.sidebar.slider("Uncertain threshold (lower edge of the borderline band)", 0.30, 0.95,
                                  config.UNCERTAIN_THRESHOLD, 0.01, key="uncertain",
                                  help="Between this and the acceptance threshold -> UNCERTAIN. "
                                       "Below it -> UNKNOWN. Set equal to the acceptance threshold to disable the band.")
    if uncertain > accept:
        st.sidebar.error("The uncertain threshold cannot be above the acceptance threshold; using the acceptance "
                         "threshold for both (UNCERTAIN band disabled).")
        uncertain = accept
    max_attempts = int(st.sidebar.number_input("Maximum verification attempts", 1, 5,
                                               config.MAX_VERIFICATION_ATTEMPTS, key="max_attempts"))
    st.sidebar.caption(
        f"Calibrated defaults: accept {config.ACCEPT_THRESHOLD:.2f}, uncertain {config.UNCERTAIN_THRESHOLD:.2f} "
        "(chosen on LFW validation data; see the Evaluation tab). The **similarity score** is the cosine "
        "similarity between face embeddings. It is *not* a probability or a calibrated confidence.")
    st.sidebar.divider()
    st.sidebar.metric("Enrolled people", db.num_identities)
    st.sidebar.metric("Stored reference embeddings", db.num_embeddings)
    st.sidebar.caption(f"Model: {config.EMBEDDING_MODEL_NAME} ({config.EMBEDDING_DIM}-D) with MTCNN detection. "
                       "Everything runs locally; no photos are stored, only embeddings.")
    return accept, uncertain, max_attempts


# ----------------------------------------------------------------------------- enroll
def enroll_tab(system: FaceRecognitionSystem, accept: float, uncertain: float) -> None:
    st.header("Enroll a person")
    st.write("Upload 1 to 10 photos of **one** person. Each photo must contain exactly one clear face. "
             f"{config.RECOMMENDED_ENROLL_IMAGES}+ photos with different lighting/pose give better recognition.")
    with st.form("enroll_form"):
        c1, c2 = st.columns(2)
        person_id = c1.text_input("Person ID", placeholder="emp001",
                                  help="Letters, digits, '_', '-', '.'; unique per person.")
        name = c2.text_input("Full name", placeholder="Anjali Rao")
        files = st.file_uploader("Face photos", type=config.ALLOWED_UPLOAD_EXTENSIONS, accept_multiple_files=True)
        add_existing = st.checkbox("Add photos to an existing identity (same ID and name)")
        consent = st.checkbox("This person has agreed to have their face enrolled")
        submitted = st.form_submit_button("Enroll", type="primary")
    if not submitted:
        return
    if not consent:
        st.error("Enrollment needs the person's consent (biometric data).")
        return
    try:
        with st.spinner("Detecting faces and computing embeddings ..."):
            report = system.enroll(person_id, name, [(f.name, f) for f in files or []],
                                   add_to_existing=add_existing, accept_threshold=accept,
                                   uncertain_threshold=uncertain)
    except EnrollmentError as exc:
        st.error(str(exc))
        return
    except DatabaseError as exc:
        st.error(f"The embeddings could not be saved: {exc}")
        return
    except PIPELINE_ERRORS as exc:
        st.error(f"Processing failed: {exc}")
        return

    if report.success:
        st.success(f"Stored {report.stored_count} new reference embedding(s) for **{report.name}** "
                   f"(ID `{report.person_id}`). Total references: {report.total_references}.")
    else:
        st.error("Nothing was enrolled: none of the photos passed the checks below.")
    for w in report.warnings:
        st.warning(w)
    for outcome in report.outcomes:
        c1, c2 = st.columns([1, 6])
        if outcome.face_crop is not None:
            c1.image(outcome.face_crop, caption="model input", width=96)
        (c2.success if outcome.accepted else c2.error)(f"**{outcome.label}**: {outcome.message}")


# ----------------------------------------------------------------------------- identify
def reset_identification() -> None:
    for k in ("id_results", "id_image", "id_thresholds", "verification", "verification_face"):
        st.session_state.pop(k, None)


def annotated(image, results):
    faces = [{"box": r.face.box, "color": BOX_COLORS_BGR[r.match.decision],
              "label": f"{r.index}: {r.match.identity_label} {r.match.similarity:.2f}"} for r in results]
    return cv2_to_pil(draw_faces(pil_to_cv2(image), faces))


def results_table(results) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r.match
        rows.append({
            "Face": r.index,
            "Result": m.identity_label,
            "Decision state": m.decision,
            "Similarity score": round(m.similarity, 3),
            "Recognition threshold": m.accept_threshold,
            "Uncertain threshold": m.uncertain_threshold,
            "Closest enrolled candidate": f"{m.candidate_name} ({m.candidate_id})" if m.candidate_id else "-",
            "Notes": "; ".join(r.warnings),
        })
    return pd.DataFrame(rows)


def start_session(face_result, max_attempts: int) -> None:
    m = face_result.match
    session = VerificationSession(m.accept_threshold, m.uncertain_threshold, max_attempts)
    session.record(m)  # the borderline result is attempt 1
    st.session_state.verification = session
    st.session_state.verification_face = face_result.index


def identify_tab(system: FaceRecognitionSystem, accept: float, uncertain: float, max_attempts: int) -> None:
    st.header("Identify faces")
    if system.database.num_identities == 0:
        st.info("Nobody is enrolled yet, so every face will be UNKNOWN. Enroll people first.")
    uploaded = st.file_uploader("Photo to identify (it may contain several faces)",
                                type=config.ALLOWED_UPLOAD_EXTENSIONS, key="identify_upload")
    if uploaded is None:
        reset_identification()
        return

    photo_key = hashlib.sha1(uploaded.getvalue()).hexdigest()
    if st.session_state.get("id_photo_key") != photo_key:  # a new photo starts a new verification session
        reset_identification()
        st.session_state.id_photo_key = photo_key

    if st.button("Identify", type="primary"):
        reset_identification()
        try:
            image = load_image(uploaded)
            with st.spinner("Detecting and identifying faces ..."):
                results = system.identify(image, accept, uncertain)
        except PIPELINE_ERRORS as exc:
            st.error(str(exc))
            return
        st.session_state.id_results, st.session_state.id_image = results, image
        st.session_state.id_thresholds = (accept, uncertain)
        uncertain_faces = [r for r in results if r.match.decision == UNCERTAIN]
        if len(results) == 1 and uncertain_faces:
            start_session(results[0], max_attempts)  # single borderline face: re-verification starts automatically

    results = st.session_state.get("id_results")
    if results is None:
        return
    image = st.session_state.id_image
    if not results:
        st.warning("No face detected. Try a clearer, well-lit, front-facing photo.")
        st.image(image, width=480)
        return
    if st.session_state.id_thresholds != (accept, uncertain):
        st.info("Thresholds changed since this photo was identified. Press **Identify** again to apply them.")

    c1, c2 = st.columns([3, 2])
    c1.image(annotated(image, results), caption=f"{len(results)} face(s) detected "
             "(green = RECOGNIZED, orange = UNCERTAIN, red = UNKNOWN)", width="stretch")
    with c2:
        for r in results:
            m = r.match
            msg = (f"**Face {r.index}: {m.identity_label}**  \nSimilarity score: **{m.similarity:.3f}**  \n"
                   f"Recognition threshold: {m.accept_threshold:.2f} (uncertain band starts at "
                   f"{m.uncertain_threshold:.2f})  \n{m.reason}")
            {RECOGNIZED: st.success, UNCERTAIN: st.warning, UNKNOWN: st.error}[m.decision](msg)
            for w in r.warnings:
                st.caption(f"Warning: {w}")
    st.dataframe(results_table(results), hide_index=True, width="stretch")

    uncertain_faces = [r for r in results if r.match.decision == UNCERTAIN]
    if uncertain_faces and "verification" not in st.session_state:
        st.subheader("Borderline faces")
        choice = st.selectbox("Re-verify which face?", [r.index for r in uncertain_faces],
                              format_func=lambda i: f"Face {i}")
        if st.button("Start re-verification for this face"):
            start_session(next(r for r in uncertain_faces if r.index == choice), max_attempts)
            st.rerun()

    if "verification" in st.session_state:
        verification_panel(system)


def verification_panel(system: FaceRecognitionSystem) -> None:
    session: VerificationSession = st.session_state.verification
    st.subheader(f"Re-verification of face {st.session_state.verification_face}")
    st.caption(f"Session thresholds (fixed for the whole session): accept {session.accept_threshold:.2f}, "
               f"uncertain {session.uncertain_threshold:.2f}. Maximum attempts: {session.max_attempts}.")
    st.progress(session.attempts_used / session.max_attempts,
                text=f"Attempts used: {session.attempts_used} of {session.max_attempts}")
    st.dataframe(pd.DataFrame([{"Attempt": a.attempt_number, "Decision": a.decision,
                                "Similarity score": round(a.similarity, 3),
                                "Closest candidate": a.candidate_name or "-"} for a in session.attempts]),
                 hide_index=True)

    if session.is_complete:
        # UNKNOWN and UNABLE_TO_VERIFY both mean "not recognized".
        (st.success if session.final_decision == RECOGNIZED else st.error)(session.status_message())
        if st.button("Start a new verification"):
            reset_identification()
            st.rerun()
        return

    n = session.next_attempt_number
    st.warning(
        f"**Attempt {n} of {session.max_attempts}.** The last similarity score "
        f"({session.attempts[-1].similarity:.3f}) is in the borderline band "
        f"[{session.uncertain_threshold:.2f}, {session.accept_threshold:.2f}), so the system will neither "
        "accept nor reject on this evidence. Please upload **another photo of the same person** for "
        "re-verification. You may re-upload the same photo, but it will give the same score because the "
        "pipeline is deterministic; a different photo (better light, facing the camera) is more useful.")
    photo = st.file_uploader(f"Photo for attempt {n}", type=config.ALLOWED_UPLOAD_EXTENSIONS,
                             key=f"reverify_{session.session_id}_{n}")
    if st.button(f"Re-verify (attempt {n} of {session.max_attempts})", disabled=photo is None, type="primary"):
        try:
            image = load_image(photo)
            with st.spinner("Running detection, embedding and matching ..."):
                result = system.verify_single(image, session.accept_threshold, session.uncertain_threshold)
        except (NoFaceError, MultipleFacesError, *PIPELINE_ERRORS) as exc:
            st.error(f"{exc} This upload did not use up an attempt.")
            return
        session.record(result.match)
        st.rerun()
    if st.button("Cancel verification"):
        reset_identification()
        st.rerun()


# ----------------------------------------------------------------------------- database
def database_tab(db: FaceDatabase) -> None:
    st.header("Enrolled people")
    identities = db.get_all_identities()
    if not identities:
        st.info("The database is empty.")
        return
    st.dataframe(pd.DataFrame([{"Person ID": pid, "Name": rec["name"], "Reference embeddings": len(rec["embeddings"]),
                                "Enrolled": rec.get("created_at", ""), "Updated": rec.get("updated_at", "")}
                               for pid, rec in identities.items()]), hide_index=True, width="stretch")
    st.caption(f"Stored at `embeddings/{db.db_path.name}`: embeddings only, no photos.")
    c1, c2 = st.columns(2)
    with c1:
        pid = st.selectbox("Remove a person", list(identities), format_func=lambda p: f"{identities[p]['name']} ({p})")
        if st.button("Remove selected person"):
            try:
                db.remove_person(pid)
            except DatabaseError as exc:
                st.error(str(exc))
            else:
                reset_identification()
                st.rerun()
    with c2:
        confirm = st.checkbox("I understand this deletes every enrolled person")
        if st.button("Clear the whole database", disabled=not confirm):
            try:
                db.clear_database()
            except DatabaseError as exc:
                st.error(str(exc))
            else:
                reset_identification()
                st.rerun()


# ----------------------------------------------------------------------------- evaluation
def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def evaluation_tab() -> None:
    st.header("Evaluation (measured results only)")
    test_dir = config.RESULTS_DIR / "lfw_test"
    summary_file = test_dir / "summary.json"
    if not summary_file.exists():
        st.info("No evaluation results found yet. Generate them with:\n\n"
                "```\npython scripts/prepare_lfw.py\n"
                "python -m src.evaluation --data data/lfw_eval/val --calibrate --out results/lfw_val\n"
                "python -m src.evaluation --data data/lfw_eval/test --thresholds-from "
                "results/lfw_val/calibration.json --out results/lfw_test\n```")
        return
    s = json.loads(summary_file.read_text(encoding="utf-8"))
    m = s["operating_point"]
    st.write(f"Dataset: **LFW test split** ({s['gallery']['identities_enrolled']} enrolled people, "
             f"{m['known']['count']} photos of enrolled people, {m['unknown']['count']} photos of "
             f"never-enrolled people). Thresholds were chosen on a *different* set of people: "
             f"accept {m['accept_threshold']:.2f}, uncertain {m['uncertain_threshold']:.2f}. "
             "These numbers describe LFW (celebrity news photos), not your own camera or users.")
    st.markdown("**One photo per attempt**")
    c = st.columns(4)
    c[0].metric("Recognized", pct(m["known"]["correctly_recognized"]),
                help="Enrolled people recognized as the correct person from a single photo.")
    c[1].metric("Unknown rejected", pct(m["unknown"]["correctly_rejected_unknown"]),
                help="Never-enrolled people correctly returned as UNKNOWN.")
    c[2].metric("False acceptance", pct(m["unknown"]["falsely_accepted"]),
                help="Never-enrolled people wrongly RECOGNIZED as someone (FAR).")
    c[3].metric("Misidentified", pct(m["known"]["misidentified_as_other_person"]),
                help="Enrolled people RECOGNIZED as a different enrolled person.")
    sk, su = s["sessions"]["known"], s["sessions"]["unknown"]
    st.markdown(f"**After re-verification (up to {s['max_attempts']} photos)**")
    c = st.columns(4)
    c[0].metric("Recognized", pct(sk["after_reverification"].get("correctly_recognized", 0)),
                help="Enrolled people recognized correctly within the allowed attempts.")
    c[1].metric("Unable to verify", pct(sk["after_reverification"].get("unable_to_verify", 0)),
                help="Enrolled people who stayed borderline on every attempt.")
    c[2].metric("False acceptance", pct(su["after_reverification"].get("falsely_accepted", 0)),
                help="Never-enrolled people accepted on any attempt. Retries give them extra chances, "
                     "so this is higher than the single-photo rate.")
    c[3].metric("Equal error rate", pct(s["equal_error_rate"]["FAR"]),
                help=f"Single threshold where FAR = FRR (at {s['equal_error_rate']['threshold']:.2f}).")
    for plot in s.get("plots", []):
        if (test_dir / plot).exists():
            st.image(str(test_dir / plot), width="stretch")
    report = test_dir / "report.md"
    if report.exists():
        with st.expander("Full report"):
            st.markdown(report.read_text(encoding="utf-8"))
    group = config.RESULTS_DIR / "group_test" / "summary.json"
    if group.exists():
        with st.expander("Critical group-image false-acceptance test"):
            st.json(json.loads(group.read_text(encoding="utf-8")))


# ----------------------------------------------------------------------------- main
def main() -> None:
    st.title("Face Recognition Identification System")
    st.caption("Enroll people, then identify faces with UNKNOWN rejection and bounded re-verification of "
               "borderline results. Free, open-source, runs on CPU.")
    try:
        detector, embedder = load_models()
    except ModelInitError as exc:
        st.error(f"Model initialisation failed: {exc}")
        st.stop()
    try:
        db = load_database()
    except DatabaseError as exc:
        st.error(f"The embedding database could not be loaded: {exc}")
        if st.button("Move the damaged file aside and start with an empty database"):
            backup = backup_corrupt_database(config.DEFAULT_DB_PATH)
            load_database.clear()
            st.success(f"Damaged file kept as {backup.name}.")
            st.rerun()
        st.stop()

    system = FaceRecognitionSystem(db, detector, embedder)
    accept, uncertain, max_attempts = sidebar_settings(db)
    tabs = st.tabs(["Enroll Person", "Identify Face", "Enrolled People", "Evaluation"])
    with tabs[0]:
        enroll_tab(system, accept, uncertain)
    with tabs[1]:
        identify_tab(system, accept, uncertain, max_attempts)
    with tabs[2]:
        database_tab(db)
    with tabs[3]:
        evaluation_tab()


if __name__ == "__main__":
    main()
