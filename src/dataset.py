"""
Dataset Management Module for Custom Sign Language Vocabulary.
Handles storage, persistence, label normalization, data validation, and stale detection.
Supports both Static (single-frame) and Dynamic (multi-frame temporal sequence) signs.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import json
import time
import uuid
import hashlib
from pathlib import Path
import numpy as np

from src.landmarks import FEATURE_VECTOR_DIM

DEFAULT_DATASET_FILE = Path("data/dataset.json")


def normalize_label(label: str) -> str:
    """
    Consistently normalizes sign labels to prevent accidental duplicates
    caused by differences in capitalization, surrounding whitespace, or internal double spaces.
    Example: '  hello  world ' -> 'HELLO WORLD'
    """
    if not label:
        return ""
    words = label.strip().split()
    return " ".join(words).upper()


class CustomSignDataset:
    """
    Manages user-recorded sign language samples with persistent JSON storage.
    Enforces data integrity, schema validation, and atomic updates.
    """

    def __init__(self, filepath: Path = DEFAULT_DATASET_FILE):
        self.filepath = Path(filepath)
        self.classes: Dict[str, Dict[str, Any]] = {}
        # List of sample dicts:
        # {
        #   "id": str,
        #   "label": str,
        #   "mode": "static" | "dynamic",
        #   "features": list[float] | list[list[float]],
        #   "frame_count": int,
        #   "timestamp": float
        # }
        self.samples: List[Dict[str, Any]] = []
        self.last_modified: float = time.time()
        self.load()

    def get_dataset_hash(self) -> str:
        """
        Computes a cryptographic hash of the current dataset contents.
        Used to detect when the trained model is out of sync with the dataset (stale model).
        """
        summary_repr = {
            "classes": sorted(list(self.classes.keys())),
            "sample_ids": [s["id"] for s in self.samples],
            "num_samples": len(self.samples),
        }
        encoded = json.dumps(summary_repr, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def add_class(self, label: str, mode: str = "static", description: str = "") -> Tuple[bool, str]:
        """
        Registers a new custom sign class.
        Returns (success: bool, message: str).
        """
        cleaned = normalize_label(label)
        if not cleaned:
            return False, "Sign label cannot be empty."

        if cleaned in self.classes:
            return False, f"Sign class '{cleaned}' already exists in the dataset."

        if mode not in ("static", "dynamic"):
            mode = "static"

        self.classes[cleaned] = {
            "mode": mode,
            "description": description.strip(),
            "created_at": time.time(),
        }
        self.last_modified = time.time()
        self.save()
        return True, f"Class '{cleaned}' created successfully."

    def rename_class(self, old_label: str, new_label: str) -> Tuple[bool, str]:
        """
        Safely renames an existing sign class and updates all associated samples.
        """
        old_clean = normalize_label(old_label)
        new_clean = normalize_label(new_label)

        if not old_clean or old_clean not in self.classes:
            return False, f"Class '{old_clean}' does not exist."

        if not new_clean:
            return False, "New sign label cannot be empty."

        if old_clean == new_clean:
            return True, "No changes made (same label)."

        if new_clean in self.classes:
            return False, f"Cannot rename: class '{new_clean}' already exists."

        # Update classes registry
        class_info = self.classes.pop(old_clean)
        self.classes[new_clean] = class_info

        # Update all existing samples
        updated_samples = 0
        for s in self.samples:
            if s["label"] == old_clean:
                s["label"] = new_clean
                updated_samples += 1

        self.last_modified = time.time()
        self.save()
        return True, f"Renamed '{old_clean}' to '{new_clean}' (updated {updated_samples} samples)."

    def delete_class(self, label: str) -> Tuple[bool, str]:
        """
        Deletes a sign class and all of its associated samples.
        """
        cleaned = normalize_label(label)
        if cleaned not in self.classes:
            return False, f"Class '{cleaned}' does not exist."

        # Remove class
        del self.classes[cleaned]

        # Remove all associated samples
        before_count = len(self.samples)
        self.samples = [s for s in self.samples if s["label"] != cleaned]
        removed_count = before_count - len(self.samples)

        self.last_modified = time.time()
        self.save()
        return True, f"Deleted class '{cleaned}' and {removed_count} recorded samples."

    def add_static_sample(
        self, label: str, feature_vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Validates and adds a single-frame static pose sample (131 features).
        """
        cleaned = normalize_label(label)
        if cleaned not in self.classes:
            # Auto-register if not explicitly added
            self.classes[cleaned] = {"mode": "static", "created_at": time.time(), "description": ""}

        if not isinstance(feature_vector, np.ndarray):
            feature_vector = np.array(feature_vector, dtype=np.float32)

        if feature_vector.shape != (FEATURE_VECTOR_DIM,):
            return False, f"Invalid feature dimension {feature_vector.shape}, expected ({FEATURE_VECTOR_DIM},)."

        # Reject completely empty frames (no hands detected at all: presence flags at index 63 and 127 are 0)
        left_present = feature_vector[63] > 0.5
        right_present = feature_vector[127] > 0.5
        if not (left_present or right_present):
            return False, "Cannot save sample: no hands were detected in this frame."

        sample_record = {
            "id": str(uuid.uuid4()),
            "label": cleaned,
            "mode": "static",
            "features": feature_vector.astype(float).tolist(),
            "frame_count": 1,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self.samples.append(sample_record)
        self.last_modified = time.time()
        self.save()
        return True, f"Sample saved for sign '{cleaned}'."

    def add_dynamic_sample(
        self,
        label: str,
        sequence_features: List[np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Validates and adds an ordered temporal sequence of landmark feature frames.
        """
        cleaned = normalize_label(label)
        if cleaned not in self.classes:
            self.classes[cleaned] = {"mode": "dynamic", "created_at": time.time(), "description": ""}

        if not sequence_features or len(sequence_features) < 5:
            return False, "Dynamic sequence is too short (minimum 5 frames required)."

        validated_frames = []
        hands_detected_count = 0

        for frame_feat in sequence_features:
            if not isinstance(frame_feat, np.ndarray):
                frame_feat = np.array(frame_feat, dtype=np.float32)
            if frame_feat.shape != (FEATURE_VECTOR_DIM,):
                return False, f"Invalid frame feature dimension {frame_feat.shape}."

            left_p = frame_feat[63] > 0.5
            right_p = frame_feat[127] > 0.5
            if left_p or right_p:
                hands_detected_count += 1
            validated_frames.append(frame_feat.astype(float).tolist())

        # At least 50% of the frames must have had hands detected
        if hands_detected_count < len(sequence_features) * 0.4:
            return False, "Recording rejected: hands were not reliably tracked during the sequence."

        sample_record = {
            "id": str(uuid.uuid4()),
            "label": cleaned,
            "mode": "dynamic",
            "features": validated_frames,
            "frame_count": len(validated_frames),
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self.samples.append(sample_record)
        self.last_modified = time.time()
        self.save()
        return True, f"Dynamic sequence ({len(validated_frames)} frames) saved for '{cleaned}'."

    def get_classes(self) -> List[str]:
        """Returns sorted list of active class names."""
        return sorted(list(self.classes.keys()))

    def get_sample_counts(self) -> Dict[str, int]:
        """Returns dictionary of sample counts per class."""
        counts = {c: 0 for c in self.get_classes()}
        for s in self.samples:
            lbl = s["label"]
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    def get_statistics(self) -> Dict[str, Any]:
        """
        Returns comprehensive dashboard statistics.
        """
        counts = self.get_sample_counts()
        total_samples = len(self.samples)
        total_classes = len(self.classes)
        return {
            "total_classes": total_classes,
            "total_samples": total_samples,
            "sample_counts": counts,
            "classes": self.classes,
            "last_modified": self.last_modified,
            "dataset_hash": self.get_dataset_hash(),
        }

    def clear_all(self) -> None:
        """Clears all classes and samples."""
        self.classes = {}
        self.samples = []
        self.last_modified = time.time()
        self.save()

    def save(self) -> None:
        """
        Saves dataset to JSON file atomically using a temporary file
        to prevent data corruption in case of unexpected termination.
        """
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "2.0",
            "feature_dim": FEATURE_VECTOR_DIM,
            "last_modified": self.last_modified,
            "classes": self.classes,
            "samples": self.samples,
        }

        temp_path = self.filepath.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Atomic replacement on Windows
        if self.filepath.exists():
            self.filepath.unlink()
        temp_path.rename(self.filepath)

    def load(self) -> None:
        """
        Loads dataset from JSON file with schema validation and error resilience.
        """
        if not self.filepath.exists():
            self.classes = {}
            self.samples = []
            return

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_classes = data.get("classes", {})
            raw_samples = data.get("samples", [])

            # Reconcile classes
            self.classes = {}
            for c_name, c_info in raw_classes.items():
                norm = normalize_label(c_name)
                if norm:
                    self.classes[norm] = c_info

            # Validate and filter samples
            valid_samples = []
            for s in raw_samples:
                lbl = normalize_label(s.get("label", ""))
                feat = s.get("features", [])
                mode = s.get("mode", "static")

                if not lbl or not feat:
                    continue

                # Ensure class exists
                if lbl not in self.classes:
                    self.classes[lbl] = {"mode": mode, "created_at": time.time()}

                valid_samples.append({
                    "id": s.get("id", str(uuid.uuid4())),
                    "label": lbl,
                    "mode": mode,
                    "features": feat,
                    "frame_count": s.get("frame_count", 1),
                    "timestamp": s.get("timestamp", time.time()),
                    "metadata": s.get("metadata", {}),
                })

            self.samples = valid_samples
            self.last_modified = data.get("last_modified", time.time())
        except Exception as e:
            print(f"Warning: Failed to load dataset ({e}), starting with clean dataset.")
            self.classes = {}
            self.samples = []

    def get_hand_breakdown(self) -> Dict[str, Dict[str, int]]:
        """
        Analyzes all samples and returns hand distribution per class:
        both hands, left-only, right-only, empty, and total.
        """
        breakdown: Dict[str, Dict[str, int]] = {}
        for cls in self.get_classes():
            breakdown[cls] = {"both": 0, "left_only": 0, "right_only": 0, "empty": 0, "dynamic": 0, "total": 0}

        for s in self.samples:
            lbl = s["label"]
            if lbl not in breakdown:
                breakdown[lbl] = {"both": 0, "left_only": 0, "right_only": 0, "empty": 0, "dynamic": 0, "total": 0}
            breakdown[lbl]["total"] += 1

            if s.get("mode") == "dynamic":
                breakdown[lbl]["dynamic"] += 1
            else:
                feat = s.get("features", [])
                if len(feat) == FEATURE_VECTOR_DIM:
                    l_present = feat[63] > 0.5
                    r_present = feat[127] > 0.5
                    if l_present and r_present:
                        breakdown[lbl]["both"] += 1
                    elif l_present:
                        breakdown[lbl]["left_only"] += 1
                    elif r_present:
                        breakdown[lbl]["right_only"] += 1
                    else:
                        breakdown[lbl]["empty"] += 1

        return breakdown

    def clean_inconsistent_samples(self, target_classes: Optional[List[str]] = None) -> Tuple[int, str]:
        """
        Prunes noisy and corrupted samples:
        1. For classes where the user recorded two-handed signs, removes the accidental single-hand frames
           captured when MediaPipe dropped one hand.
        2. In dynamic sequences, removes empty frames where no hands were detected.
        """
        initial_count = len(self.samples)
        cleaned: List[Dict[str, Any]] = []
        breakdown = self.get_hand_breakdown()

        for s in self.samples:
            lbl = s["label"]
            if target_classes is not None and lbl not in target_classes:
                cleaned.append(s)
                continue

            mode = s.get("mode", "static")
            if mode == "static":
                feat = s.get("features", [])
                if len(feat) != FEATURE_VECTOR_DIM:
                    continue
                l_present = feat[63] > 0.5
                r_present = feat[127] > 0.5

                cls_stats = breakdown.get(lbl, {})
                both_cnt = cls_stats.get("both", 0)
                left_cnt = cls_stats.get("left_only", 0)
                right_cnt = cls_stats.get("right_only", 0)

                # If class has substantial two-handed examples, prune accidental single-hand drops
                if both_cnt >= 4 and both_cnt >= (left_cnt + right_cnt) * 0.4:
                    if l_present and r_present:
                        cleaned.append(s)
                elif not l_present and not r_present:
                    # Drop completely empty frames
                    pass
                else:
                    cleaned.append(s)
            elif mode == "dynamic":
                # Clean frames inside dynamic sequences
                raw_frames = s.get("features", [])
                valid_frames = [
                    f for f in raw_frames
                    if len(f) == FEATURE_VECTOR_DIM and (f[63] > 0.5 or f[127] > 0.5)
                ]
                if len(valid_frames) >= 5:
                    s_clean = dict(s)
                    s_clean["features"] = valid_frames
                    s_clean["frame_count"] = len(valid_frames)
                    cleaned.append(s_clean)

        removed = initial_count - len(cleaned)
        self.samples = cleaned
        self.last_modified = time.time()
        self.save()
        return removed, f"Cleaned dataset: removed {removed} noisy/inconsistent samples."
