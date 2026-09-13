"""UI smoke tests: run the Streamlit app headlessly (streamlit.testing) against a temporary database.

They check that every tab renders without exceptions, that invalid settings are handled, that the
Evaluation tab shows the measured numbers from results/, and that removing a person works.
(File uploads cannot be simulated by streamlit.testing; the upload flows are covered by
test_integration.py through the same pipeline code the app calls.)
"""
import json

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import config
from src.database import FaceDatabase


@pytest.fixture
def run_app(tmp_path, monkeypatch):
    """Return a function that runs the app with an isolated database (never the real one)."""
    monkeypatch.setattr(config, "DEFAULT_DB_PATH", tmp_path / "database.json")
    st.cache_resource.clear()  # the database object is cached; make the app open the temp file

    def _run() -> AppTest:
        at = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=180).run()
        if any("Model initialisation failed" in e.value for e in at.error):
            pytest.skip("pretrained weights unavailable")
        return at

    yield _run
    st.cache_resource.clear()


def metric(at: AppTest, label: str):
    return [m.value for m in at.metric if m.label == label]


def test_app_renders_all_tabs_without_errors(run_app):
    at = run_app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Enroll Person", "Identify Face", "Enrolled People", "Evaluation"]
    assert metric(at, "Enrolled people") == ["0"]
    assert metric(at, "Recognition threshold") == [f"≥ {config.ACCEPT_THRESHOLD:.2f}"]


def test_uncertain_threshold_above_accept_is_handled(run_app):
    at = run_app()
    at.sidebar.slider(key="uncertain").set_value(0.90).run()
    assert not at.exception
    assert any("cannot be above" in e.value for e in at.sidebar.error)


def test_evaluation_tab_shows_measured_numbers_only(run_app):
    at = run_app()
    summary = config.RESULTS_DIR / "lfw_test" / "summary.json"
    if not summary.exists():
        assert any("No evaluation results" in i.value for i in at.info)
        return
    s = json.loads(summary.read_text(encoding="utf-8"))
    far = 100 * s["operating_point"]["unknown"]["falsely_accepted"]
    assert f"{far:.2f}%" in metric(at, "False acceptance")


def test_enrolled_people_listed_and_removable(run_app):
    FaceDatabase().add_embeddings("emp001", "Test Person", [np.eye(config.EMBEDDING_DIM)[0]])
    at = run_app()
    assert metric(at, "Enrolled people") == ["1"]
    next(b for b in at.button if b.label == "Remove selected person").click().run()
    assert not at.exception
    assert metric(at, "Enrolled people") == ["0"]
    assert FaceDatabase().num_identities == 0
