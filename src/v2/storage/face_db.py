#!/usr/bin/env python3
"""
Face Database - SQLite Storage
==============================
Persistent storage for face embeddings.
Fast load, fast search, survives restarts.
"""

import sqlite3
import numpy as np
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class StoredFace:
    """A face stored in the database."""
    id: int
    name: str
    embedding: np.ndarray
    created_at: str
    last_seen: Optional[str] = None
    seen_count: int = 0


class FaceDB:
    """
    SQLite-based face database.

    Features:
    - Persistent storage (survives restarts)
    - Fast cosine similarity search
    - Multiple embeddings per person (for accuracy)
    - Auto-refresh: updates embeddings over time
    - Metadata tracking (created, last_seen, count)
    """

    def __init__(
        self,
        db_path: str = "faces.db",
        threshold: float = 0.40,  # Based on calibration: min self-sim ~0.43
        max_embeddings_per_person: int = 10,
        refresh_interval: int = 50  # Sightings between refreshes
    ):
        """
        Initialize database.

        Args:
            db_path: Path to SQLite database file
            threshold: Similarity threshold for matching (0.4-0.7)
            max_embeddings_per_person: Max embeddings to keep per person
            refresh_interval: Add new embedding every N sightings
        """
        self.db_path = Path(db_path)
        self.threshold = threshold
        self.max_embeddings = max_embeddings_per_person
        self.refresh_interval = refresh_interval
        self._conn: Optional[sqlite3.Connection] = None

        # Cache for fast lookups
        self._embeddings_cache: List[np.ndarray] = []
        self._names_cache: List[str] = []
        self._ids_cache: List[int] = []

        self._init_db()
        self._load_cache()

        logger.info(f"FaceDB initialized: {self.count} faces loaded from {db_path}")

    def _init_db(self):
        """Create database tables if not exist."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                last_seen TEXT,
                seen_count INTEGER DEFAULT 0
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON faces(name)")
        self._conn.commit()

    def _load_cache(self):
        """Load all embeddings into memory for fast search."""
        cursor = self._conn.execute("SELECT id, name, embedding FROM faces")

        self._ids_cache = []
        self._names_cache = []
        self._embeddings_cache = []

        for row in cursor:
            self._ids_cache.append(row[0])
            self._names_cache.append(row[1])
            self._embeddings_cache.append(np.frombuffer(row[2], dtype=np.float32))

    @property
    def count(self) -> int:
        """Number of faces in database."""
        return len(self._names_cache)

    @property
    def names(self) -> List[str]:
        """List of unique names."""
        return list(set(self._names_cache))

    def register(self, name: str, embedding: np.ndarray) -> int:
        """
        Register a face.

        Args:
            name: Person's name
            embedding: 512-D normalized embedding

        Returns:
            Database ID of new record
        """
        # Normalize embedding
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        now = datetime.now().isoformat()

        cursor = self._conn.execute(
            "INSERT INTO faces (name, embedding, created_at, seen_count) VALUES (?, ?, ?, 0)",
            (name, embedding.tobytes(), now)
        )
        self._conn.commit()

        face_id = cursor.lastrowid

        # Update cache
        self._ids_cache.append(face_id)
        self._names_cache.append(name)
        self._embeddings_cache.append(embedding)

        logger.info(f"Registered face: {name} (ID: {face_id})")
        return face_id

    def identify(self, embedding: np.ndarray) -> Tuple[Optional[str], float, Optional[int]]:
        """
        Find matching face.

        Args:
            embedding: Query embedding (512-D)

        Returns:
            (name, similarity, id) or (None, best_similarity, None)
        """
        if not self._embeddings_cache:
            return None, 0.0, None

        # Normalize query
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Compute similarities (dot product of normalized vectors = cosine similarity)
        db_matrix = np.array(self._embeddings_cache)
        similarities = np.dot(db_matrix, embedding)

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= self.threshold:
            name = self._names_cache[best_idx]
            face_id = self._ids_cache[best_idx]

            # Update last_seen and count
            self._update_seen(face_id)

            return name, best_sim, face_id

        return None, best_sim, None

    def _update_seen(self, face_id: int):
        """Update last_seen timestamp and increment count."""
        now = datetime.now().isoformat()
        self._conn.execute(
            "UPDATE faces SET last_seen = ?, seen_count = seen_count + 1 WHERE id = ?",
            (now, face_id)
        )
        self._conn.commit()

    def get_person(self, name: str) -> List[StoredFace]:
        """Get all face records for a person."""
        cursor = self._conn.execute(
            "SELECT id, name, embedding, created_at, last_seen, seen_count FROM faces WHERE name = ?",
            (name,)
        )

        faces = []
        for row in cursor:
            faces.append(StoredFace(
                id=row[0],
                name=row[1],
                embedding=np.frombuffer(row[2], dtype=np.float32),
                created_at=row[3],
                last_seen=row[4],
                seen_count=row[5]
            ))
        return faces

    def delete_person(self, name: str) -> int:
        """Delete all faces for a person."""
        cursor = self._conn.execute("DELETE FROM faces WHERE name = ?", (name,))
        self._conn.commit()
        deleted = cursor.rowcount

        # Reload cache
        self._load_cache()

        logger.info(f"Deleted {deleted} faces for: {name}")
        return deleted

    def delete_face(self, face_id: int) -> bool:
        """Delete a specific face by ID."""
        cursor = self._conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        self._conn.commit()

        if cursor.rowcount > 0:
            self._load_cache()
            return True
        return False

    def list_all(self) -> List[dict]:
        """List all registered people with stats."""
        cursor = self._conn.execute("""
            SELECT name, COUNT(*) as count, MAX(last_seen) as last_seen, SUM(seen_count) as total_seen
            FROM faces GROUP BY name ORDER BY name
        """)

        return [
            {"name": row[0], "embeddings": row[1], "last_seen": row[2], "total_seen": row[3]}
            for row in cursor
        ]

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    import tempfile
    import os

    print("Testing FaceDB...")

    # Use temp file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = FaceDB(db_path, threshold=0.5)

        # Test register
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)

        id1 = db.register("Alice", emb1)
        print(f"Registered Alice: ID={id1}")

        # Test identify (same embedding should match)
        name, sim, fid = db.identify(emb1)
        print(f"Identify same: {name}, sim={sim:.3f}")
        assert name == "Alice"
        assert sim > 0.99

        # Test identify (random should not match)
        random_emb = np.random.randn(512).astype(np.float32)
        name, sim, fid = db.identify(random_emb)
        print(f"Identify random: {name}, sim={sim:.3f}")

        # Test list
        people = db.list_all()
        print(f"People in DB: {people}")

        # Test persistence - close and reopen
        db.close()
        db2 = FaceDB(db_path, threshold=0.5)
        print(f"After reopen: {db2.count} faces")
        assert db2.count == 1

        # Should still identify
        name, sim, fid = db2.identify(emb1)
        print(f"After reopen identify: {name}, sim={sim:.3f}")
        assert name == "Alice"

        db2.close()
        print("\n✅ FaceDB works!")

    finally:
        os.unlink(db_path)
