"""Bounded re-verification of borderline (UNCERTAIN) results.

A session starts with one match result. RECOGNIZED or UNKNOWN end it immediately. UNCERTAIN
asks for another photo, which goes through the full pipeline again (detect -> embed -> match).
After max_attempts uncertain results the session ends as UNABLE_TO_VERIFY, which must be
treated as "not recognized". The session refuses further attempts once it is complete, so an
endless loop is impossible.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from . import config
from .recognizer import RECOGNIZED, UNCERTAIN, UNKNOWN, MatchResult, validate_thresholds

UNABLE_TO_VERIFY = "UNABLE_TO_VERIFY"


class VerificationSessionClosed(RuntimeError):
    """An attempt was submitted to a session that has already finished."""


@dataclass
class AttemptRecord:
    attempt_number: int
    decision: str
    similarity: float
    candidate_id: str | None
    candidate_name: str | None


class VerificationSession:
    def __init__(self, accept_threshold: float, uncertain_threshold: float,
                 max_attempts: int = config.MAX_VERIFICATION_ATTEMPTS):
        if int(max_attempts) != max_attempts or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer.")
        validate_thresholds(accept_threshold, uncertain_threshold)
        self.session_id = uuid.uuid4().hex
        # Thresholds are frozen for the whole session so every attempt is judged the same way.
        self.accept_threshold = accept_threshold
        self.uncertain_threshold = uncertain_threshold
        self.max_attempts = int(max_attempts)
        self.attempts: list[AttemptRecord] = []
        self.final_decision: str | None = None
        self.final_result: MatchResult | None = None

    @property
    def is_complete(self) -> bool:
        return self.final_decision is not None

    @property
    def attempts_used(self) -> int:
        return len(self.attempts)

    @property
    def attempts_remaining(self) -> int:
        return 0 if self.is_complete else self.max_attempts - self.attempts_used

    @property
    def next_attempt_number(self) -> int | None:
        return None if self.is_complete else self.attempts_used + 1

    def record(self, result: MatchResult) -> str | None:
        """Add one attempt's result. Returns the final decision if the session is now complete,
        otherwise None (meaning: ask for another photo)."""
        if self.is_complete:
            raise VerificationSessionClosed(
                f"Verification already finished with {self.final_decision}; start a new session.")
        if (result.accept_threshold, result.uncertain_threshold) != (self.accept_threshold,
                                                                     self.uncertain_threshold):
            raise ValueError("Attempt was scored with different thresholds than this session.")
        if result.decision not in (RECOGNIZED, UNCERTAIN, UNKNOWN):
            raise ValueError(f"Unknown decision '{result.decision}'.")

        self.attempts.append(AttemptRecord(self.attempts_used + 1, result.decision, result.similarity,
                                           result.candidate_id, result.candidate_name))
        if result.decision in (RECOGNIZED, UNKNOWN):  # conclusive
            self.final_decision, self.final_result = result.decision, result
        elif self.attempts_used >= self.max_attempts:  # still uncertain and out of attempts
            self.final_decision, self.final_result = UNABLE_TO_VERIFY, result
        return self.final_decision

    def status_message(self) -> str:
        if self.final_decision == RECOGNIZED:
            r = self.final_result
            return (f"RECOGNIZED as {r.candidate_name} on attempt {self.attempts_used} of "
                    f"{self.max_attempts} (similarity {r.similarity:.3f}).")
        if self.final_decision == UNKNOWN:
            return f"UNKNOWN: attempt {self.attempts_used} was clearly below the borderline band."
        if self.final_decision == UNABLE_TO_VERIFY:
            return (f"UNABLE TO VERIFY: all {self.max_attempts} attempts were borderline. "
                    "The person is NOT recognized.")
        last = self.attempts[-1] if self.attempts else None
        detail = f" (last similarity {last.similarity:.3f})" if last else ""
        return (f"Borderline result{detail}. Please upload another photo for re-verification: "
                f"attempt {self.next_attempt_number} of {self.max_attempts}.")
