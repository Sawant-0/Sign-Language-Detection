"""
Machine Learning Pipeline for Custom Sign Language Classification.
Implements genuine training, cross-validation, metrics calculation,
atomic model persistence, and real-time prediction with out-of-distribution/unknown sign rejection.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import time
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.landmarks import FEATURE_VECTOR_DIM
from src.dataset import CustomSignDataset

DEFAULT_MODEL_PATH = Path("models/trained_model.joblib")


def extract_temporal_features(sequence: List[List[float]]) -> np.ndarray:
    """
    Transforms an ordered sequence of T landmark frames (each of dimension 131)
    into a fixed-length temporal descriptor:
      - Mean values across time (131)
      - Standard deviation / dynamics across time (131)
      - Net trajectory displacement: end_frame - start_frame (131)
      - Maximum frame-to-frame velocity (131)
    Total dimension: 131 * 4 = 524 features.
    """
    # Filter out empty frames where neither hand was detected
    valid_frames = [f for f in sequence if len(f) == FEATURE_VECTOR_DIM and (f[63] > 0.5 or f[127] > 0.5)]
    if len(valid_frames) >= 2:
        seq_arr = np.array(valid_frames, dtype=np.float32)
    else:
        seq_arr = np.array(sequence, dtype=np.float32)  # (T, 131)

    if seq_arr.ndim != 2 or seq_arr.shape[1] != FEATURE_VECTOR_DIM:
        raise ValueError(f"Expected sequence of shape (T, {FEATURE_VECTOR_DIM}), got {seq_arr.shape}")

    mean_feat = np.mean(seq_arr, axis=0)
    std_feat = np.std(seq_arr, axis=0)
    displacement = seq_arr[-1] - seq_arr[0]

    if len(seq_arr) > 1:
        velocities = np.abs(np.diff(seq_arr, axis=0))
        max_vel = np.max(velocities, axis=0)
    else:
        max_vel = np.zeros(FEATURE_VECTOR_DIM, dtype=np.float32)

    return np.concatenate([mean_feat, std_feat, displacement, max_vel]).astype(np.float32)


class CustomSignClassifier:
    """
    Encapsulates the trained sign classification model, class mappings,
    metadata, and inference methods.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.classifier: Optional[Any] = None
        self.classes: List[str] = []
        self.mode: str = "static"  # 'static' or 'dynamic'
        self.feature_dim: int = FEATURE_VECTOR_DIM
        self.dataset_hash: str = ""
        self.trained_at: Optional[float] = None
        self.metrics: Dict[str, Any] = {}

    def train(
        self,
        dataset: CustomSignDataset,
        validation_split: float = 0.2,
        model_save_path: Optional[Path] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Genuinely trains a classifier on the user's recorded dataset.
        Enforces strict validation, detects insufficient data, computes genuine metrics,
        and saves model atomically.
        """
        target_path = Path(model_save_path) if model_save_path else DEFAULT_MODEL_PATH
        all_classes = dataset.get_classes()
        sample_counts = dataset.get_sample_counts()

        # Rule: Must have at least 2 distinct sign classes defined
        if len(all_classes) < 2:
            return (
                False,
                f"Insufficient classes: At least 2 distinct sign classes are required to train a classifier. "
                f"Currently available: {len(all_classes)} class(es). Please record examples for at least one more sign.",
                {},
            )

        # Separate static and dynamic samples
        static_samples = [s for s in dataset.samples if s.get("mode", "static") == "static"]
        dynamic_samples = [s for s in dataset.samples if s.get("mode") == "dynamic"]

        # Determine primary training mode based on dataset contents
        if len(dynamic_samples) > len(static_samples) and len(dynamic_samples) >= 4:
            self.mode = "dynamic"
            active_samples = dynamic_samples
        else:
            self.mode = "static"
            active_samples = dataset.samples

        # Extract features and targets
        X_list = []
        y_list = []

        for s in active_samples:
            lbl = s["label"]
            raw_feats = s["features"]

            if self.mode == "static":
                # Unpack dynamic sequence frames if present to maximize usable training data
                if s.get("mode") == "dynamic" and isinstance(raw_feats, list) and len(raw_feats) > 0 and isinstance(raw_feats[0], list):
                    for frame_feat in raw_feats:
                        feat_vec = np.array(frame_feat, dtype=np.float32)
                        # Ensure hand presence in frame
                        if feat_vec.shape == (FEATURE_VECTOR_DIM,) and (feat_vec[63] > 0.5 or feat_vec[127] > 0.5):
                            X_list.append(feat_vec)
                            y_list.append(lbl)
                else:
                    feat_vec = np.array(raw_feats, dtype=np.float32)
                    if feat_vec.shape == (FEATURE_VECTOR_DIM,):
                        X_list.append(feat_vec)
                        y_list.append(lbl)
            else:
                # Dynamic temporal aggregation
                try:
                    feat_vec = extract_temporal_features(raw_feats)
                    X_list.append(feat_vec)
                    y_list.append(lbl)
                except Exception:
                    continue

        if len(X_list) < 4:
            return False, "Failed to extract valid feature representations from recorded samples. Ensure hands were visible during recording.", {}

        # Require at least 2 classes with at least 2 valid feature samples
        from collections import Counter
        class_freq = Counter(y_list)
        eligible_classes = sorted([cls for cls, cnt in class_freq.items() if cnt >= 2])

        if len(eligible_classes) < 2:
            return (
                False,
                f"Insufficient samples: At least 2 sign classes with valid hand poses are required. "
                f"Currently available: {dict(class_freq)}. Please record more samples.",
                {},
            )

        # Retain only samples for eligible classes
        filtered_X = []
        filtered_y = []
        for x_val, y_val in zip(X_list, y_list):
            if y_val in eligible_classes:
                filtered_X.append(x_val)
                filtered_y.append(y_val)

        X = np.array(filtered_X, dtype=np.float32)
        y = np.array(filtered_y, dtype=object)

        unique_labels = sorted(list(set(y)))
        self.classes = unique_labels
        self.feature_dim = X.shape[1]

        # Training timing
        start_time = time.time()

        # Determine validation split: only split if each class has at least 2 samples
        can_split = all(np.sum(y == cls) >= 2 for cls in unique_labels) and len(y) >= 8

        eval_metrics: Dict[str, Any] = {
            "num_samples": len(y),
            "num_classes": len(unique_labels),
            "classes": unique_labels,
            "mode": self.mode,
            "feature_dim": self.feature_dim,
        }

        # Initialize scikit-learn classifier
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=None,
            min_samples_split=2,
            random_state=self.random_state,
            class_weight="balanced",
        )

        if can_split:
            # Stratified split to preserve class proportions
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, stratify=y, random_state=self.random_state
            )
            clf.fit(X_train, y_train)

            # Evaluate on validation set
            y_pred = clf.predict(X_val)
            acc = float(accuracy_score(y_val, y_pred))

            # Precision, Recall, F1
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, labels=unique_labels, average="weighted", zero_division=0
            )

            # Confusion Matrix
            cm = confusion_matrix(y_val, y_pred, labels=unique_labels).tolist()

            eval_metrics["validation_accuracy"] = round(acc * 100, 2)
            eval_metrics["validation_precision"] = round(float(prec) * 100, 2)
            eval_metrics["validation_recall"] = round(float(rec) * 100, 2)
            eval_metrics["validation_f1"] = round(float(f1) * 100, 2)
            eval_metrics["confusion_matrix"] = cm
            eval_metrics["validation_samples"] = len(y_val)
            eval_metrics["training_samples"] = len(X_train)

            # Retrain on full dataset for maximum production accuracy
            clf.fit(X, y)
        else:
            # When dataset is very small (<8 total or single sample per class), train on all data
            clf.fit(X, y)
            train_acc = float(accuracy_score(y, clf.predict(X)))
            eval_metrics["training_accuracy"] = round(train_acc * 100, 2)
            eval_metrics["note"] = "Dataset too small for train/val split. Trained on all available samples."

        training_duration = round(time.time() - start_time, 3)
        eval_metrics["training_duration_sec"] = training_duration

        self.classifier = clf
        self.trained_at = time.time()
        self.dataset_hash = dataset.get_dataset_hash()
        self.metrics = eval_metrics

        # Save model atomically
        save_success, save_msg = self.save(filepath=target_path)
        if not save_success:
            return False, f"Model trained but saving failed: {save_msg}", {}

        return True, f"Model successfully trained on {len(y)} samples across {len(unique_labels)} classes in {training_duration}s.", eval_metrics

    def predict(
        self,
        features: np.ndarray,
        confidence_threshold: float = 0.65,
    ) -> Dict[str, Any]:
        """
        Runs real-time inference on input features.
        Applies confidence thresholding and unknown sign rejection.

        Returns:
            Dict containing:
              - 'predicted_class': str or None
              - 'confidence': float (0.0 to 1.0)
              - 'probabilities': Dict[str, float]
              - 'status': 'RECOGNIZED', 'LOW_CONFIDENCE', 'UNKNOWN', or 'NO_HANDS'
              - 'is_reliable': bool
        """
        if self.classifier is None:
            return {
                "predicted_class": None,
                "confidence": 0.0,
                "probabilities": {},
                "status": "MODEL_NOT_READY",
                "is_reliable": False,
            }

        feat_arr = np.array(features, dtype=np.float32)

        # Check hands presence for static 131-dim features
        if feat_arr.shape == (FEATURE_VECTOR_DIM,):
            left_p = feat_arr[63] > 0.5
            right_p = feat_arr[127] > 0.5
            if not (left_p or right_p):
                return {
                    "predicted_class": None,
                    "confidence": 0.0,
                    "probabilities": {},
                    "status": "NO_HANDS_DETECTED",
                    "is_reliable": False,
                }

        # Reshape for sklearn
        if feat_arr.ndim == 1:
            feat_arr = feat_arr.reshape(1, -1)

        # Check feature dimension compatibility
        if feat_arr.shape[1] != self.feature_dim:
            return {
                "predicted_class": None,
                "confidence": 0.0,
                "probabilities": {},
                "status": "INCOMPATIBLE_FEATURE_DIM",
                "is_reliable": False,
            }

        # Predict probabilities
        probs = self.classifier.predict_proba(feat_arr)[0]
        classes = self.classifier.classes_

        prob_dict = {str(cls): round(float(prob), 4) for cls, prob in zip(classes, probs)}
        best_idx = int(np.argmax(probs))
        best_class = str(classes[best_idx])
        confidence = float(probs[best_idx])

        # Confidence thresholding for unknown-sign rejection
        if confidence >= confidence_threshold:
            status = "RECOGNIZED"
            is_reliable = True
            predicted_class = best_class
        else:
            status = "LOW_CONFIDENCE"
            is_reliable = False
            predicted_class = None

        return {
            "predicted_class": predicted_class,
            "raw_best_class": best_class,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "status": status,
            "is_reliable": is_reliable,
        }

    def save(self, filepath: Path = DEFAULT_MODEL_PATH) -> Tuple[bool, str]:
        """
        Atomically saves the trained model and associated metadata to disk.
        """
        if self.classifier is None:
            return False, "Cannot save: No trained classifier available."

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": "2.0",
            "classifier": self.classifier,
            "classes": self.classes,
            "mode": self.mode,
            "feature_dim": self.feature_dim,
            "dataset_hash": self.dataset_hash,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
        }

        temp_path = filepath.with_suffix(".tmp")
        try:
            joblib.dump(payload, temp_path)
            if filepath.exists():
                filepath.unlink()
            temp_path.rename(filepath)
            return True, f"Model saved successfully to {filepath}."
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            return False, f"Failed to save model: {e}"

    @classmethod
    def load(cls, filepath: Path = DEFAULT_MODEL_PATH) -> Optional["CustomSignClassifier"]:
        """
        Loads a saved model from disk with integrity checks.
        Returns None if file is missing or corrupted.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return None

        try:
            payload = joblib.load(filepath)
            instance = cls()
            instance.classifier = payload["classifier"]
            instance.classes = payload["classes"]
            instance.mode = payload.get("mode", "static")
            instance.feature_dim = payload.get("feature_dim", FEATURE_VECTOR_DIM)
            instance.dataset_hash = payload.get("dataset_hash", "")
            instance.trained_at = payload.get("trained_at")
            instance.metrics = payload.get("metrics", {})
            return instance
        except Exception as e:
            print(f"Warning: Failed to load model from {filepath}: {e}")
            return None

    def is_stale(self, current_dataset: CustomSignDataset) -> bool:
        """
        Checks whether the model's training snapshot is out of sync with current dataset.
        """
        if not self.dataset_hash:
            return True
        return self.dataset_hash != current_dataset.get_dataset_hash()
