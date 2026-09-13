"""Similarity matching and the decision policy.

For a query embedding q and every enrolled identity P with reference embeddings r_1..r_k:
    score(P) = max_i cosine_similarity(q, r_i)        (best matching reference photo)
The identity with the highest score is the *candidate*. The candidate is only accepted if its
score clears the acceptance threshold; being the closest enrolled person is never enough.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config

RECOGNIZED = "RECOGNIZED"
UNCERTAIN = "UNCERTAIN"
UNKNOWN = "UNKNOWN"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(a, b) = (a . b) / (||a|| * ||b||), in [-1, 1]. Returns 0.0 for a zero vector."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Embedding size mismatch: {a.shape} vs {b.shape}")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def validate_thresholds(accept_threshold: float, uncertain_threshold: float) -> None:
    if not -1.0 <= uncertain_threshold <= accept_threshold <= 1.0:
        raise ValueError(
            "Thresholds must satisfy -1 <= uncertain_threshold <= accept_threshold <= 1 "
            f"(got uncertain={uncertain_threshold}, accept={accept_threshold})."
        )


def decide(similarity: float, accept_threshold: float, uncertain_threshold: float) -> str:
    """The decision policy.
        similarity >= accept_threshold                        -> RECOGNIZED
        uncertain_threshold <= similarity < accept_threshold  -> UNCERTAIN
        similarity < uncertain_threshold                      -> UNKNOWN
    Setting uncertain_threshold == accept_threshold disables the UNCERTAIN band.
    """
    validate_thresholds(accept_threshold, uncertain_threshold)
    if similarity >= accept_threshold:
        return RECOGNIZED
    if similarity >= uncertain_threshold:
        return UNCERTAIN
    return UNKNOWN


@dataclass
class MatchResult:
    decision: str                   # RECOGNIZED / UNCERTAIN / UNKNOWN
    similarity: float               # cosine similarity of the best candidate (nan if database empty)
    candidate_id: str | None        # best-matching enrolled identity (NOT necessarily accepted)
    candidate_name: str | None
    accept_threshold: float
    uncertain_threshold: float
    top_candidates: list[tuple[str, str, float]] = field(default_factory=list)  # (id, name, sim)
    reason: str = ""

    @property
    def is_recognized(self) -> bool:
        return self.decision == RECOGNIZED

    @property
    def identity_label(self) -> str:
        """What to show as 'the answer': a name only when recognized."""
        return self.candidate_name if self.decision == RECOGNIZED else self.decision


class FaceRecognizer:
    def __init__(self, database, accept_threshold: float = config.ACCEPT_THRESHOLD,
                 uncertain_threshold: float = config.UNCERTAIN_THRESHOLD):
        validate_thresholds(accept_threshold, uncertain_threshold)
        self.database = database
        self.accept_threshold = accept_threshold
        self.uncertain_threshold = uncertain_threshold

    def rank_identities(self, query_embedding: np.ndarray) -> list[tuple[str, str, float]]:
        """Score every enrolled identity; returns [(person_id, name, similarity)] best first.

        All references are compared at once: rows of the matrix and the query are unit vectors,
        so matrix @ query is the cosine similarity with every reference (same formula as
        cosine_similarity, vectorised). Each identity keeps its best reference score.
        """
        matrix, owners = self.database.get_embedding_matrix()
        if matrix.shape[0] == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if q.shape[0] != matrix.shape[1]:
            raise ValueError(f"Query embedding has {q.shape[0]} values, expected {matrix.shape[1]}.")
        norm = np.linalg.norm(q)
        if norm == 0 or not np.all(np.isfinite(q)):
            raise ValueError("Query embedding is invalid (zero vector or NaN).")
        sims = matrix @ (q / norm)

        best: dict[str, float] = {}
        for pid, s in zip(owners, sims):
            if pid not in best or s > best[pid]:
                best[pid] = float(s)
        identities = self.database.get_all_identities()
        ranked = [(pid, identities[pid]["name"], s) for pid, s in best.items()]
        ranked.sort(key=lambda t: t[2], reverse=True)
        return ranked

    def identify(self, query_embedding: np.ndarray, accept_threshold: float | None = None,
                 uncertain_threshold: float | None = None, top_k: int = 3) -> MatchResult:
        accept = self.accept_threshold if accept_threshold is None else accept_threshold
        uncertain = self.uncertain_threshold if uncertain_threshold is None else uncertain_threshold
        validate_thresholds(accept, uncertain)

        ranked = self.rank_identities(query_embedding)
        if not ranked:
            return MatchResult(UNKNOWN, float("nan"), None, None, accept, uncertain, [],
                               reason="No identities are enrolled, so nobody can be recognized.")

        pid, name, sim = ranked[0]
        decision = decide(sim, accept, uncertain)
        if decision == RECOGNIZED:
            reason = f"Best match {name} scored {sim:.3f} >= acceptance threshold {accept:.2f}."
        elif decision == UNCERTAIN:
            reason = (f"Best match {name} scored {sim:.3f}: inside the borderline band "
                      f"[{uncertain:.2f}, {accept:.2f}). Another photo is needed to decide.")
        else:
            reason = (f"Closest enrolled identity ({name}) scored only {sim:.3f} < {uncertain:.2f}; "
                      "this face does not match anyone enrolled.")
        return MatchResult(decision, sim, pid, name, accept, uncertain, ranked[:top_k], reason)
