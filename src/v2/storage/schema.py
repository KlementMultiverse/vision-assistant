#!/usr/bin/env python3
"""
Vision Assistant Database Schema
=================================
Full schema for persons, embeddings, and visits.

Tables:
  - persons: Everyone we've seen (family/friends/public)
  - embeddings: Multiple face vectors per person
  - visits: Every continuous observation logged
"""

import sqlite3
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class GroupType(Enum):
    FAMILY = "family"
    FRIENDS = "friends"
    PUBLIC = "public"


class PersonStatus(Enum):
    ACTIVE = "active"
    BLOCKED = "blocked"


@dataclass
class Person:
    """A person in the database."""
    id: int
    name: str
    display_name: Optional[str] = None
    group_type: str = "public"  # family, friends, public
    role: Optional[str] = None  # daughter, son, delivery_guy, school_friend, work_colleague, etc.
    status: str = "active"
    visit_count: int = 0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Embedding:
    """A face embedding."""
    id: int
    person_id: int
    embedding: np.ndarray
    confidence: float
    captured_at: str
    lighting: Optional[str] = None


@dataclass
class Visit:
    """A single continuous observation."""
    id: int
    person_id: int
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: int = 0
    frame_count: int = 0
    best_frame_path: Optional[str] = None
    avg_confidence: float = 0.0
    vision_context: Optional[str] = None
    bot_interaction: Optional[str] = None


class VisionDB:
    """
    Full database for vision assistant.

    Handles persons, embeddings, and visits with proper relationships.
    """

    def __init__(
        self,
        db_path: str = "vision.db",
        match_threshold: float = 0.40,  # Based on calibration data
        min_detection_confidence: float = 0.5,  # Discard faces below this
        min_good_frames: int = 2,  # Need at least N good frames to identify
    ):
        self.db_path = Path(db_path)
        self.match_threshold = match_threshold
        self.min_detection_confidence = min_detection_confidence
        self.min_good_frames = min_good_frames
        self._conn: Optional[sqlite3.Connection] = None

        # Cache for fast matching
        self._embeddings_cache: List[np.ndarray] = []
        self._person_ids_cache: List[int] = []
        self._embedding_ids_cache: List[int] = []

        # Counter for unknown naming
        self._unknown_counter: int = 0

        self._init_db()
        self._load_cache()

        logger.info(f"VisionDB initialized: {self.person_count} persons, {self.embedding_count} embeddings")

    def _init_db(self):
        """Create tables."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Persons table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                display_name TEXT,
                group_type TEXT DEFAULT 'public',
                role TEXT,
                status TEXT DEFAULT 'active',
                visit_count INTEGER DEFAULT 0,
                first_seen TEXT,
                last_seen TEXT,
                notes TEXT
            )
        """)

        # Add role column if not exists (for existing databases)
        try:
            self._conn.execute("ALTER TABLE persons ADD COLUMN role TEXT")
            self._conn.commit()
        except:
            pass  # Column already exists

        # Embeddings table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                confidence FLOAT DEFAULT 0.0,
                captured_at TEXT NOT NULL,
                lighting TEXT,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
            )
        """)

        # Visits table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration_seconds INTEGER DEFAULT 0,
                frame_count INTEGER DEFAULT 0,
                best_frame_path TEXT,
                avg_confidence FLOAT DEFAULT 0.0,
                vision_context TEXT,
                bot_interaction TEXT,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
            )
        """)

        # Indexes
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_persons_group ON persons(group_type)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_visits_person ON visits(person_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_visits_time ON visits(start_time)")

        self._conn.commit()

        # Get current unknown counter
        cursor = self._conn.execute(
            "SELECT name FROM persons WHERE name LIKE 'unknown_%' ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row:
            try:
                self._unknown_counter = int(row[0].split('_')[1])
            except:
                self._unknown_counter = 0

    def _load_cache(self):
        """Load all embeddings for fast matching."""
        cursor = self._conn.execute("""
            SELECT e.id, e.person_id, e.embedding
            FROM embeddings e
            JOIN persons p ON e.person_id = p.id
            WHERE p.status = 'active'
        """)

        self._embeddings_cache = []
        self._person_ids_cache = []
        self._embedding_ids_cache = []

        for row in cursor:
            self._embedding_ids_cache.append(row[0])
            self._person_ids_cache.append(row[1])
            self._embeddings_cache.append(np.frombuffer(row[2], dtype=np.float32))

    @property
    def person_count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM persons")
        return cursor.fetchone()[0]

    @property
    def embedding_count(self) -> int:
        return len(self._embeddings_cache)

    # =========================================================================
    # PERSON OPERATIONS
    # =========================================================================

    def create_unknown(
        self,
        embeddings_with_conf: List[Tuple[np.ndarray, float]],
        vision_context: Optional[str] = None
    ) -> Tuple[int, str]:
        """
        Create a new unknown person with BEST embedding only.

        Args:
            embeddings_with_conf: List of (embedding, confidence) from visit
            vision_context: What vision model saw

        Returns:
            (person_id, name) e.g., (5, "unknown_5")

        Logic:
            - Filter to confidence > 0.5
            - Pick the BEST one (highest confidence)
            - Store only that one (keep unknowns light)
        """
        # Filter quality embeddings
        quality = [(emb, conf) for emb, conf in embeddings_with_conf
                   if conf >= self.min_detection_confidence]

        if not quality:
            # No good frames, use best of what we have
            if embeddings_with_conf:
                quality = [max(embeddings_with_conf, key=lambda x: x[1])]
            else:
                return None, None

        # Pick best
        best_emb, best_conf = max(quality, key=lambda x: x[1])

        self._unknown_counter += 1
        name = f"unknown_{self._unknown_counter}"
        now = datetime.now().isoformat()

        # Create person
        cursor = self._conn.execute("""
            INSERT INTO persons (name, group_type, status, visit_count, first_seen, last_seen, notes)
            VALUES (?, 'public', 'active', 1, ?, ?, ?)
        """, (name, now, now, vision_context))
        person_id = cursor.lastrowid

        # Add ONLY the best embedding
        self._add_embedding(person_id, best_emb, best_conf)

        self._conn.commit()
        self._load_cache()

        logger.info(f"Created unknown: {name} (ID: {person_id}, conf: {best_conf:.3f})")
        return person_id, name

    def add_embedding_if_quality(
        self,
        person_id: int,
        embedding: np.ndarray,
        confidence: float,
        min_similarity: float = 0.4,
        max_per_person: int = 10
    ) -> bool:
        """
        Add embedding only if it's quality AND confirms same person.

        Used for known persons (family/friends) to build rich profile.

        Args:
            person_id: Person to add embedding to
            embedding: New embedding
            confidence: Detection confidence
            min_similarity: Must match existing embeddings above this
            max_per_person: Limit embeddings per person

        Returns:
            True if added, False if rejected
        """
        # Check detection confidence
        if confidence < self.min_detection_confidence:
            return False

        # Normalize
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Get existing embeddings for this person
        person_emb_indices = [
            i for i, pid in enumerate(self._person_ids_cache) if pid == person_id
        ]

        if person_emb_indices:
            # Check similarity to existing (confirms same person)
            existing = [self._embeddings_cache[i] for i in person_emb_indices]
            similarities = [float(np.dot(embedding, e)) for e in existing]
            max_sim = max(similarities)

            if max_sim < min_similarity:
                logger.debug(f"Rejected: similarity {max_sim:.3f} < {min_similarity}")
                return False

        # Add it
        self.add_embedding(person_id, embedding, confidence, max_per_person=max_per_person)
        logger.info(f"Added quality embedding to person {person_id} (conf: {confidence:.3f})")
        return True

    def get_person(self, person_id: int) -> Optional[Person]:
        """Get person by ID."""
        cursor = self._conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
        row = cursor.fetchone()
        if row:
            return Person(**dict(row))
        return None

    def get_person_by_name(self, name: str) -> Optional[Person]:
        """Get person by name."""
        cursor = self._conn.execute("SELECT * FROM persons WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            return Person(**dict(row))
        return None

    def rename_person(self, person_id: int, new_name: str,
                      display_name: Optional[str] = None) -> bool:
        """Rename a person (e.g., unknown_5 → 'delivery_guy')."""
        try:
            self._conn.execute("""
                UPDATE persons SET name = ?, display_name = ? WHERE id = ?
            """, (new_name, display_name or new_name, person_id))
            self._conn.commit()
            logger.info(f"Renamed person {person_id} to {new_name}")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Name {new_name} already exists")
            return False

    def set_group(self, person_id: int, group: str) -> bool:
        """Change person's group (family/friends/public)."""
        if group not in ['family', 'friends', 'public']:
            return False
        self._conn.execute("UPDATE persons SET group_type = ? WHERE id = ?", (group, person_id))
        self._conn.commit()
        return True

    def set_role(self, person_id: int, role: str) -> bool:
        """Set person's role (daughter, son, delivery_guy, school_friend, work_colleague, etc.)."""
        self._conn.execute("UPDATE persons SET role = ? WHERE id = ?", (role, person_id))
        self._conn.commit()
        return True

    def tag_person(self, person_id: int, name: str, group: str, role: Optional[str] = None) -> bool:
        """Quick tag: set name, group, and role in one call."""
        try:
            self._conn.execute("""
                UPDATE persons SET name = ?, display_name = ?, group_type = ?, role = ?
                WHERE id = ?
            """, (name, name, group, role, person_id))
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Name already exists

    def block_person(self, person_id: int) -> bool:
        """Block a person (alerts only, no greeting)."""
        self._conn.execute("UPDATE persons SET status = 'blocked' WHERE id = ?", (person_id,))
        self._conn.commit()
        self._load_cache()  # Remove from matching cache
        return True

    def list_persons(self, group: Optional[str] = None) -> List[Person]:
        """List all persons, optionally filtered by group."""
        if group:
            cursor = self._conn.execute(
                "SELECT * FROM persons WHERE group_type = ? ORDER BY last_seen DESC", (group,)
            )
        else:
            cursor = self._conn.execute("SELECT * FROM persons ORDER BY last_seen DESC")
        return [Person(**dict(row)) for row in cursor]

    def merge_persons(self, keep_id: int, merge_id: int) -> bool:
        """Merge two persons (e.g., discovered unknown_3 = unknown_5)."""
        # Move embeddings
        self._conn.execute(
            "UPDATE embeddings SET person_id = ? WHERE person_id = ?",
            (keep_id, merge_id)
        )
        # Move visits
        self._conn.execute(
            "UPDATE visits SET person_id = ? WHERE person_id = ?",
            (keep_id, merge_id)
        )
        # Update visit count
        self._conn.execute("""
            UPDATE persons SET visit_count = (
                SELECT COUNT(*) FROM visits WHERE person_id = ?
            ) WHERE id = ?
        """, (keep_id, keep_id))
        # Delete merged person
        self._conn.execute("DELETE FROM persons WHERE id = ?", (merge_id,))
        self._conn.commit()
        self._load_cache()
        logger.info(f"Merged person {merge_id} into {keep_id}")
        return True

    # =========================================================================
    # EMBEDDING OPERATIONS
    # =========================================================================

    def _add_embedding(self, person_id: int, embedding: np.ndarray,
                       confidence: float = 0.0, lighting: str = None):
        """Add embedding to person (internal)."""
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        now = datetime.now().isoformat()
        self._conn.execute("""
            INSERT INTO embeddings (person_id, embedding, confidence, captured_at, lighting)
            VALUES (?, ?, ?, ?, ?)
        """, (person_id, embedding.tobytes(), confidence, now, lighting))

    def add_embedding(self, person_id: int, embedding: np.ndarray,
                      confidence: float = 0.0, lighting: str = None,
                      max_per_person: int = 10) -> int:
        """
        Add new embedding to person.

        Limits embeddings per person, removing oldest if exceeded.
        """
        # Check count
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE person_id = ?", (person_id,)
        )
        count = cursor.fetchone()[0]

        # Remove oldest if at limit
        if count >= max_per_person:
            self._conn.execute("""
                DELETE FROM embeddings WHERE id = (
                    SELECT id FROM embeddings WHERE person_id = ?
                    ORDER BY captured_at ASC LIMIT 1
                )
            """, (person_id,))

        self._add_embedding(person_id, embedding, confidence, lighting)
        self._conn.commit()
        self._load_cache()

        return self.get_embedding_count(person_id)

    def get_embedding_count(self, person_id: int) -> int:
        """Get number of embeddings for a person."""
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE person_id = ?", (person_id,)
        )
        return cursor.fetchone()[0]

    # =========================================================================
    # MATCHING
    # =========================================================================

    def identify(self, embedding: np.ndarray) -> Tuple[Optional[Person], float, Optional[int]]:
        """
        Find matching person from single embedding.

        Returns:
            (Person, similarity, embedding_id) or (None, best_sim, None)
        """
        if not self._embeddings_cache:
            return None, 0.0, None

        # Normalize
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Compute similarities
        db_matrix = np.array(self._embeddings_cache)
        similarities = np.dot(db_matrix, embedding)

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= self.match_threshold:
            person_id = self._person_ids_cache[best_idx]
            embedding_id = self._embedding_ids_cache[best_idx]
            person = self.get_person(person_id)
            return person, best_sim, embedding_id

        return None, best_sim, None

    def identify_from_batch(
        self,
        embeddings: List[Tuple[np.ndarray, float]]  # (embedding, confidence)
    ) -> Tuple[Optional[Person], float, Dict]:
        """
        Identify from multiple frames - handles real-world messy data.

        Args:
            embeddings: List of (embedding, detection_confidence) tuples
                        Include None for frames with no face detected

        Returns:
            (Person, avg_similarity, stats) or (None, 0.0, stats)

        Logic:
            1. Filter out None/low-confidence detections
            2. If not enough good frames, return unknown
            3. Match each good embedding against DB
            4. Use majority vote / average similarity
        """
        stats = {
            'total_frames': len(embeddings),
            'null_frames': 0,
            'low_confidence': 0,
            'good_frames': 0,
            'matches': {},  # person_id -> [similarities]
        }

        good_embeddings = []

        # Step 1: Filter
        for item in embeddings:
            if item is None or item[0] is None:
                stats['null_frames'] += 1
                continue

            emb, conf = item
            if conf < self.min_detection_confidence:
                stats['low_confidence'] += 1
                continue

            good_embeddings.append(emb)
            stats['good_frames'] += 1

        # Step 2: Check minimum
        if stats['good_frames'] < self.min_good_frames:
            logger.info(f"Not enough good frames: {stats['good_frames']}/{self.min_good_frames}")
            return None, 0.0, stats

        # Step 3: Match each embedding
        for emb in good_embeddings:
            person, sim, _ = self.identify(emb)
            if person:
                if person.id not in stats['matches']:
                    stats['matches'][person.id] = []
                stats['matches'][person.id].append(sim)

        # Step 4: Majority vote
        if not stats['matches']:
            return None, 0.0, stats

        # Find person with most matches
        best_person_id = max(stats['matches'], key=lambda x: len(stats['matches'][x]))
        best_sims = stats['matches'][best_person_id]

        # Require majority of good frames to match
        match_ratio = len(best_sims) / stats['good_frames']
        if match_ratio < 0.5:  # Less than half matched
            logger.info(f"No majority: {len(best_sims)}/{stats['good_frames']} matched")
            return None, 0.0, stats

        avg_sim = float(np.mean(best_sims))
        person = self.get_person(best_person_id)

        return person, avg_sim, stats

    # =========================================================================
    # VISIT OPERATIONS
    # =========================================================================

    def start_visit(self, person_id: int, vision_context: Optional[str] = None) -> int:
        """Start a new visit. Returns visit_id."""
        now = datetime.now().isoformat()

        cursor = self._conn.execute("""
            INSERT INTO visits (person_id, start_time, vision_context)
            VALUES (?, ?, ?)
        """, (person_id, now, vision_context))
        visit_id = cursor.lastrowid

        # Update person's last_seen and visit_count
        self._conn.execute("""
            UPDATE persons SET last_seen = ?, visit_count = visit_count + 1
            WHERE id = ?
        """, (now, person_id))

        self._conn.commit()
        return visit_id

    def end_visit(self, visit_id: int, frame_count: int = 0,
                  avg_confidence: float = 0.0, best_frame_path: str = None,
                  bot_interaction: str = None):
        """End a visit with final stats."""
        now = datetime.now().isoformat()

        # Get start time to calculate duration
        cursor = self._conn.execute(
            "SELECT start_time FROM visits WHERE id = ?", (visit_id,)
        )
        row = cursor.fetchone()
        if row:
            start = datetime.fromisoformat(row[0])
            end = datetime.now()
            duration = int((end - start).total_seconds())

            self._conn.execute("""
                UPDATE visits SET
                    end_time = ?, duration_seconds = ?, frame_count = ?,
                    avg_confidence = ?, best_frame_path = ?, bot_interaction = ?
                WHERE id = ?
            """, (now, duration, frame_count, avg_confidence, best_frame_path,
                  bot_interaction, visit_id))
            self._conn.commit()

    def get_visits(self, person_id: Optional[int] = None,
                   date: Optional[str] = None,
                   limit: int = 100) -> List[Dict]:
        """
        Get visits, optionally filtered.

        Args:
            person_id: Filter by person
            date: Filter by date (YYYY-MM-DD)
            limit: Max results
        """
        query = """
            SELECT v.*, p.name, p.display_name, p.group_type
            FROM visits v
            JOIN persons p ON v.person_id = p.id
            WHERE 1=1
        """
        params = []

        if person_id:
            query += " AND v.person_id = ?"
            params.append(person_id)

        if date:
            query += " AND v.start_time LIKE ?"
            params.append(f"{date}%")

        query += " ORDER BY v.start_time DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor]

    def get_today_summary(self) -> Dict:
        """Get summary of today's activity."""
        today = datetime.now().strftime("%Y-%m-%d")

        cursor = self._conn.execute("""
            SELECT
                COUNT(*) as total_visits,
                COUNT(DISTINCT person_id) as unique_visitors,
                SUM(CASE WHEN p.group_type = 'family' THEN 1 ELSE 0 END) as family_visits,
                SUM(CASE WHEN p.group_type = 'friends' THEN 1 ELSE 0 END) as friend_visits,
                SUM(CASE WHEN p.group_type = 'public' THEN 1 ELSE 0 END) as public_visits
            FROM visits v
            JOIN persons p ON v.person_id = p.id
            WHERE v.start_time LIKE ?
        """, (f"{today}%",))

        row = cursor.fetchone()
        return dict(row) if row else {}

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def auto_learn(
        self,
        person_id: int,
        embedding: np.ndarray,
        confidence: float,
        min_difference: float = 0.3,  # Only add if different enough
        max_embeddings: int = 10
    ) -> bool:
        """
        Auto-learn: Add embedding if it's different from existing ones.

        Called when person is recognized but embedding might capture
        new conditions (lighting, angle, etc.)

        Args:
            person_id: Recognized person
            embedding: Current embedding
            confidence: Detection confidence
            min_difference: Only add if max_similarity < (1 - min_difference)
            max_embeddings: Cap embeddings per person

        Returns:
            True if embedding was added
        """
        if confidence < self.min_detection_confidence:
            return False

        # Check current count
        current_count = self.get_embedding_count(person_id)
        if current_count >= max_embeddings:
            return False

        # Normalize
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Get existing embeddings for this person
        person_indices = [
            i for i, pid in enumerate(self._person_ids_cache) if pid == person_id
        ]

        if not person_indices:
            return False

        # Check similarity to existing
        existing = [self._embeddings_cache[i] for i in person_indices]
        similarities = [float(np.dot(embedding, e)) for e in existing]
        max_sim = max(similarities)

        # Only add if different enough
        if max_sim > (1 - min_difference):  # Too similar to existing
            return False

        # Add it
        self._add_embedding(person_id, embedding, confidence, lighting="auto")
        self._conn.commit()
        self._load_cache()

        logger.info(f"Auto-learned: person {person_id}, diff={1-max_sim:.3f}")
        return True

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

    print("Testing VisionDB...")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = VisionDB(db_path, match_threshold=0.5)

        # Create unknown person
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)

        person_id, name = db.create_unknown(emb1, confidence=0.85, vision_context="person at door")
        print(f"Created: {name} (ID: {person_id})")

        # Identify (should match)
        person, sim, _ = db.identify(emb1)
        print(f"Identified: {person.name}, sim={sim:.3f}")
        assert person.name == name

        # Start visit
        visit_id = db.start_visit(person_id, vision_context="wearing blue shirt")
        print(f"Started visit: {visit_id}")

        # End visit
        db.end_visit(visit_id, frame_count=50, avg_confidence=0.88)
        print("Ended visit")

        # Rename
        db.rename_person(person_id, "delivery_guy", "Delivery Person")
        person = db.get_person(person_id)
        print(f"Renamed to: {person.name}")

        # Add more embeddings
        for i in range(5):
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            count = db.add_embedding(person_id, emb, confidence=0.8 + i*0.02)
        print(f"Embeddings for person: {db.get_embedding_count(person_id)}")

        # Create another unknown
        emb2 = np.random.randn(512).astype(np.float32)
        person_id2, name2 = db.create_unknown(emb2)
        print(f"Created: {name2}")

        # List all
        print("\nAll persons:")
        for p in db.list_persons():
            print(f"  - {p.name} ({p.group_type}): {p.visit_count} visits")

        # Get visits
        print("\nRecent visits:")
        for v in db.get_visits(limit=5):
            print(f"  - {v['name']} at {v['start_time'][:19]}")

        db.close()
        print("\n✅ VisionDB works!")

    finally:
        os.unlink(db_path)
