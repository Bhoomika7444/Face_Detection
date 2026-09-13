"""Persistent local storage of enrolled identities and their face embeddings (one JSON file).

File layout (schema_version 2):
{
  "schema_version": 2,
  "embedding_model": "facenet-pytorch/InceptionResnetV1/vggface2",
  "embedding_dim": 512,
  "identities": {
    "emp001": {"name": "Rahul Sharma", "embeddings": [[512 floats], ...],
               "created_at": "...", "updated_at": "..."}
  }
}
Only embeddings are stored: no photos and no file names.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config

SCHEMA_VERSION = 2
_PERSON_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class DatabaseError(RuntimeError):
    """The database file is unreadable/invalid, or a write failed."""


def validate_person_id(person_id: str) -> str:
    person_id = (person_id or "").strip()
    if not _PERSON_ID_RE.match(person_id):
        raise ValueError(
            "Person ID must be 1-64 characters: letters, digits, '_', '-' or '.', "
            "starting with a letter or digit (e.g. emp001)."
        )
    return person_id


def validate_name(name: str) -> str:
    name = " ".join((name or "").split())  # trim and collapse whitespace
    if not 1 <= len(name) <= 100:
        raise ValueError("Name must be 1-100 characters.")
    if any(not ch.isprintable() for ch in name):
        raise ValueError("Name contains invalid (non-printable) characters.")
    return name


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def backup_corrupt_database(db_path: Path = config.DEFAULT_DB_PATH) -> Path:
    """Move an unreadable database file aside (keeps it for inspection) so a fresh one can start."""
    db_path = Path(db_path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = db_path.with_name(f"{db_path.stem}.corrupt-{stamp}{db_path.suffix}")
    os.replace(db_path, backup)
    return backup


class FaceDatabase:
    def __init__(self, db_dir: str | Path | None = None, db_name: str = "database.json",
                 embedding_dim: int = config.EMBEDDING_DIM,
                 model_name: str = config.EMBEDDING_MODEL_NAME):
        """db_dir=None uses <project>/embeddings, so the location never depends on the CWD."""
        self.db_path = Path(db_dir) / db_name if db_dir is not None else config.DEFAULT_DB_PATH
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.data: dict[str, dict] = {}
        self._matrix_cache: tuple[np.ndarray, list[str]] | None = None
        self.load_db()

    # ------------------------------------------------------------------ persistence
    def load_db(self) -> None:
        """Load from disk. A missing file means an empty database. An unreadable or invalid file
        raises DatabaseError and is left untouched (it is never silently overwritten)."""
        self._matrix_cache = None
        if not self.db_path.exists():
            self.data = {}
            return
        try:
            raw = json.loads(self.db_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatabaseError(f"Embedding database {self.db_path} is unreadable or corrupted: {exc}") from exc

        if isinstance(raw, dict) and "identities" not in raw and "schema_version" not in raw:
            identities = raw  # legacy v1 format: {person_id: {"name":..., "embeddings": [...]}}
        elif isinstance(raw, dict):
            if raw.get("embedding_model", self.model_name) != self.model_name:
                raise DatabaseError(
                    f"Database was created with model '{raw.get('embedding_model')}', but this system uses "
                    f"'{self.model_name}'. Embeddings from different models cannot be compared; re-enroll."
                )
            if int(raw.get("embedding_dim", self.embedding_dim)) != self.embedding_dim:
                raise DatabaseError("Database embedding dimension does not match the model.")
            identities = raw.get("identities", {})
        else:
            raise DatabaseError("Embedding database has an unexpected structure.")
        if not isinstance(identities, dict):
            raise DatabaseError("Embedding database 'identities' must be an object.")

        data = {}
        for pid, rec in identities.items():
            if not isinstance(rec, dict) or "name" not in rec or "embeddings" not in rec:
                raise DatabaseError(f"Identity '{pid}' is missing 'name' or 'embeddings'.")
            try:
                embs = [self._validate_embedding(e) for e in rec["embeddings"]]
            except ValueError as exc:
                raise DatabaseError(f"Identity '{pid}' has invalid embedding data: {exc}") from exc
            if not embs:
                raise DatabaseError(f"Identity '{pid}' has no embeddings.")
            data[str(pid)] = {
                "name": str(rec["name"]),
                "embeddings": [e.tolist() for e in embs],
                "created_at": rec.get("created_at", ""),
                "updated_at": rec.get("updated_at", ""),
            }
        self.data = data

    def save_db(self) -> None:
        """Atomic save: write a temp file, then os.replace(), so a crash cannot leave half a file."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "embedding_model": self.model_name,
            "embedding_dim": self.embedding_dim,
            "identities": self.data,
        }
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.db_path.parent, prefix=".db-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, separators=(",", ":"))
                os.replace(tmp, self.db_path)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        except OSError as exc:
            raise DatabaseError(f"Could not save the embedding database to {self.db_path}: {exc}") from exc

    # ------------------------------------------------------------------ validation
    def _validate_embedding(self, embedding) -> np.ndarray:
        arr = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if arr.shape[0] != self.embedding_dim:
            raise ValueError(f"expected {self.embedding_dim} values, got {arr.shape[0]}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("embedding contains NaN or infinity")
        norm = np.linalg.norm(arr)
        if norm == 0:
            raise ValueError("embedding is a zero vector")
        return np.round(arr / norm, 8)  # store unit vectors; 8 decimals is beyond float32 precision

    # ------------------------------------------------------------------ writes
    def add_embeddings(self, person_id: str, name: str, embeddings: list[np.ndarray]) -> int:
        """Add reference embeddings for a person (creating the identity if needed). Returns the
        person's total number of references. The same ID with a different name is rejected."""
        person_id, name = validate_person_id(person_id), validate_name(name)
        if not embeddings:
            raise ValueError("No embeddings to store.")
        new = [self._validate_embedding(e).tolist() for e in embeddings]
        existing = self.data.get(person_id)
        if existing and existing["name"].casefold() != name.casefold():
            raise ValueError(f"ID '{person_id}' is already enrolled as '{existing['name']}', not '{name}'.")

        previous = None if existing is None else {**existing, "embeddings": list(existing["embeddings"])}
        now = _now()
        if existing is None:
            self.data[person_id] = {"name": name, "embeddings": new, "created_at": now, "updated_at": now}
        else:
            existing["embeddings"].extend(new)
            existing["updated_at"] = now
        self._matrix_cache = None
        try:
            self.save_db()
        except DatabaseError:
            # Roll back the in-memory change so memory and disk stay consistent.
            if previous is None:
                self.data.pop(person_id, None)
            else:
                self.data[person_id] = previous
            raise
        return len(self.data[person_id]["embeddings"])

    def enroll_person(self, person_id: str, name: str, embedding: np.ndarray) -> int:
        """Backwards-compatible single-embedding enrollment."""
        return self.add_embeddings(person_id, name, [embedding])

    def remove_person(self, person_id: str) -> bool:
        if person_id not in self.data:
            return False
        removed = self.data.pop(person_id)
        self._matrix_cache = None
        try:
            self.save_db()
        except DatabaseError:
            self.data[person_id] = removed
            raise
        return True

    def clear_database(self) -> None:
        previous = self.data
        self.data, self._matrix_cache = {}, None
        try:
            self.save_db()
        except DatabaseError:
            self.data = previous
            raise

    # ------------------------------------------------------------------ reads
    def get_all_identities(self) -> dict:
        return self.data

    def get_person(self, person_id: str) -> dict | None:
        return self.data.get(person_id)

    def has_person(self, person_id: str) -> bool:
        return person_id in self.data

    @property
    def num_identities(self) -> int:
        return len(self.data)

    @property
    def num_embeddings(self) -> int:
        return sum(len(r["embeddings"]) for r in self.data.values())

    def get_embedding_matrix(self) -> tuple[np.ndarray, list[str]]:
        """All reference embeddings stacked as an (n, 512) float32 matrix (unit rows) plus the
        owner person_id of each row. Cached until the database changes."""
        if self._matrix_cache is None:
            rows, owners = [], []
            for pid, rec in self.data.items():
                for e in rec["embeddings"]:
                    rows.append(e)
                    owners.append(pid)
            matrix = (np.asarray(rows, dtype=np.float32) if rows
                      else np.zeros((0, self.embedding_dim), dtype=np.float32))
            self._matrix_cache = (matrix, owners)
        return self._matrix_cache
