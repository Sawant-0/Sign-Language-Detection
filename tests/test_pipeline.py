"""
Comprehensive Automated Test Suite for Sign Language Detection System.
Tests feature extraction dimensions, missing-hand representation, dataset CRUD,
validation rules, model training, persistence, prediction consistency,
unknown gesture rejection, and camera resource cleanup.
"""

import unittest
import tempfile
import shutil
import time
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.landmarks import (
    DualHandLandmarkExtractor,
    FEATURE_VECTOR_DIM,
    create_synthetic_features,
)
from src.dataset import CustomSignDataset, normalize_label
from src.model import CustomSignClassifier, extract_temporal_features
from src.camera import CameraManager
from src.recognizer import RealTimeRecognizer


class TestSignLanguagePipeline(unittest.TestCase):
    """Unit and Integration Test Suite."""

    def setUp(self):
        """Create a temporary directory for test artifacts."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_dataset_path = self.test_dir / "test_dataset.json"
        self.test_model_path = self.test_dir / "test_model.joblib"

    def tearDown(self):
        """Clean up temporary test artifacts."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # 1. Landmark feature extraction & expected dimensions
    def test_feature_vector_dimension(self):
        """Verify that synthetic and extracted feature vectors match exact 131 dimensions."""
        feat = create_synthetic_features(left_present=True, right_present=True)
        self.assertEqual(feat.shape, (FEATURE_VECTOR_DIM,))
        self.assertEqual(feat.dtype, np.float32)

    # 2. Missing hand representations
    def test_missing_hand_representations(self):
        """Verify presence flags and zero-padding for individual and both missing hands."""
        # Case A: Left hand only
        left_only = create_synthetic_features(left_present=True, right_present=False)
        self.assertEqual(left_only[63], 1.0)   # Left presence flag
        self.assertEqual(left_only[127], 0.0)  # Right presence flag
        self.assertTrue(np.all(left_only[64:127] == 0.0))  # Right coords zeroed
        self.assertTrue(np.all(left_only[128:131] == 0.0)) # Inter-hand zeroed

        # Case B: Right hand only
        right_only = create_synthetic_features(left_present=False, right_present=True)
        self.assertEqual(right_only[63], 0.0)
        self.assertEqual(right_only[127], 1.0)
        self.assertTrue(np.all(right_only[0:63] == 0.0))

        # Case C: Neither hand present
        neither = create_synthetic_features(left_present=False, right_present=False)
        self.assertEqual(neither[63], 0.0)
        self.assertEqual(neither[127], 0.0)
        self.assertTrue(np.all(neither == 0.0))

    # 3. Two-hand feature processing
    def test_two_hand_feature_processing(self):
        """Verify both presence indicators are set and inter-hand vector is populated."""
        both = create_synthetic_features(left_present=True, right_present=True)
        self.assertEqual(both[63], 1.0)
        self.assertEqual(both[127], 1.0)
        # Coordinates for both hands should be non-zero
        self.assertFalse(np.all(both[0:63] == 0.0))
        self.assertFalse(np.all(both[64:127] == 0.0))

    # 4. Label normalization
    def test_label_normalization(self):
        """Verify robust normalization prevents accidental duplicate classes."""
        self.assertEqual(normalize_label("  hello   world  "), "HELLO WORLD")
        self.assertEqual(normalize_label("water"), "WATER")
        self.assertEqual(normalize_label(""), "")

    # 5. Static sample collection and persistence
    def test_static_sample_collection_and_persistence(self):
        """Verify saving and reloading static pose samples from disk."""
        dataset = CustomSignDataset(self.test_dataset_path)
        ok, _ = dataset.add_class("HELLO", mode="static")
        self.assertTrue(ok)

        feat = create_synthetic_features(left_present=True, right_present=False)
        save_ok, msg = dataset.add_static_sample("HELLO", feat)
        self.assertTrue(save_ok, msg)

        # Reload from disk
        reloaded = CustomSignDataset(self.test_dataset_path)
        self.assertIn("HELLO", reloaded.get_classes())
        self.assertEqual(len(reloaded.samples), 1)
        self.assertEqual(len(reloaded.samples[0]["features"]), FEATURE_VECTOR_DIM)

    # 6. Dynamic sequence collection and persistence
    def test_dynamic_sequence_collection_and_persistence(self):
        """Verify saving and reloading dynamic temporal sequence samples."""
        dataset = CustomSignDataset(self.test_dataset_path)
        dataset.add_class("WAVE", mode="dynamic")

        seq = [create_synthetic_features(left_present=True, right_present=False) for _ in range(20)]
        save_ok, msg = dataset.add_dynamic_sample("WAVE", seq)
        self.assertTrue(save_ok, msg)

        # Reload from disk
        reloaded = CustomSignDataset(self.test_dataset_path)
        self.assertEqual(len(reloaded.samples), 1)
        self.assertEqual(reloaded.samples[0]["mode"], "dynamic")
        self.assertEqual(reloaded.samples[0]["frame_count"], 20)

    # 7. Creating, listing, renaming, and deleting custom labels
    def test_class_crud_operations(self):
        """Verify class creation, duplicate warning, renaming, and deletion."""
        dataset = CustomSignDataset(self.test_dataset_path)

        # Create
        ok, _ = dataset.add_class("WATER")
        self.assertTrue(ok)

        # Duplicate rejection
        dup_ok, _ = dataset.add_class("water")
        self.assertFalse(dup_ok)

        # Add sample to WATER
        feat = create_synthetic_features(left_present=True)
        dataset.add_static_sample("WATER", feat)

        # Rename
        rename_ok, msg = dataset.rename_class("WATER", "DRINK")
        self.assertTrue(rename_ok, msg)
        self.assertIn("DRINK", dataset.get_classes())
        self.assertNotIn("WATER", dataset.get_classes())
        self.assertEqual(dataset.samples[0]["label"], "DRINK")

        # Delete
        del_ok, _ = dataset.delete_class("DRINK")
        self.assertTrue(del_ok)
        self.assertEqual(len(dataset.get_classes()), 0)
        self.assertEqual(len(dataset.samples), 0)

    # 8. Training failure on insufficient data (<2 classes or <3 samples)
    def test_training_failure_on_insufficient_data(self):
        """Verify training properly fails with informative message on insufficient data."""
        dataset = CustomSignDataset(self.test_dataset_path)
        clf = CustomSignClassifier()

        # Case A: 0 classes
        ok, msg, _ = clf.train(dataset)
        self.assertFalse(ok)
        self.assertIn("Insufficient classes", msg)

        # Case B: Only 1 class
        dataset.add_class("YES")
        for _ in range(5):
            dataset.add_static_sample("YES", create_synthetic_features(left_present=True))
        ok, msg, _ = clf.train(dataset)
        self.assertFalse(ok)
        self.assertIn("Insufficient classes", msg)

        # Case C: 2 classes, but one class has only 1 sample (< 3)
        dataset.add_class("NO")
        dataset.add_static_sample("NO", create_synthetic_features(right_present=True))
        ok, msg, _ = clf.train(dataset)
        self.assertFalse(ok)
        self.assertIn("Insufficient samples", msg)

    # 9. Genuine training on small synthetic test dataset
    def test_genuine_model_training(self):
        """Train classifier on synthetic dataset and verify genuine evaluation metrics."""
        dataset = CustomSignDataset(self.test_dataset_path)
        dataset.add_class("YES")
        dataset.add_class("NO")

        # Add 10 samples per class with distinct features for deterministic separation
        for _ in range(10):
            f_yes = create_synthetic_features(left_present=True, right_present=False, noise_level=0.01)
            dataset.add_static_sample("YES", f_yes)

            f_no = create_synthetic_features(left_present=False, right_present=True, noise_level=0.01)
            dataset.add_static_sample("NO", f_no)

        clf = CustomSignClassifier(n_estimators=30, random_state=42)
        ok, msg, metrics = clf.train(dataset, validation_split=0.2, model_save_path=self.test_model_path)

        self.assertTrue(ok, msg)
        self.assertIn("YES", clf.classes)
        self.assertIn("NO", clf.classes)
        self.assertGreater(metrics.get("validation_accuracy", metrics.get("training_accuracy", 0)), 80.0)

    # 10. Saving, reloading, and prediction consistency
    def test_model_persistence_and_prediction(self):
        """Verify saving model to disk, loading it back, and getting consistent predictions."""
        dataset = CustomSignDataset(self.test_dataset_path)
        dataset.add_class("ALPHA")
        dataset.add_class("BETA")

        for _ in range(8):
            dataset.add_static_sample("ALPHA", create_synthetic_features(left_present=True, right_present=False))
            dataset.add_static_sample("BETA", create_synthetic_features(left_present=False, right_present=True))

        clf = CustomSignClassifier(n_estimators=20, random_state=42)
        train_ok, _, _ = clf.train(dataset, model_save_path=self.test_model_path)
        self.assertTrue(train_ok)

        save_ok, _ = clf.save(self.test_model_path)
        self.assertTrue(save_ok)

        # Reload
        reloaded = CustomSignClassifier.load(self.test_model_path)
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.classes, clf.classes)
        self.assertEqual(reloaded.feature_dim, FEATURE_VECTOR_DIM)

        # Predict using reloaded model
        test_feat = create_synthetic_features(left_present=True, right_present=False)
        pred_res = reloaded.predict(test_feat, confidence_threshold=0.50)
        self.assertIn(pred_res["status"], ["RECOGNIZED", "LOW_CONFIDENCE"])
        self.assertEqual(pred_res["predicted_class"], "ALPHA")

    # 11. Unknown and low-confidence prediction handling
    def test_unknown_and_low_confidence_rejection(self):
        """Verify confidence thresholding and empty-hand rejection."""
        dataset = CustomSignDataset(self.test_dataset_path)
        dataset.add_class("SIGN_A")
        dataset.add_class("SIGN_B")

        for _ in range(8):
            dataset.add_static_sample("SIGN_A", create_synthetic_features(left_present=True))
            dataset.add_static_sample("SIGN_B", create_synthetic_features(right_present=True))

        clf = CustomSignClassifier(n_estimators=20)
        clf.train(dataset, model_save_path=self.test_model_path)

        # Test Case 1: No hands in frame (presence flags are 0)
        empty_feat = create_synthetic_features(left_present=False, right_present=False)
        res_empty = clf.predict(empty_feat)
        self.assertEqual(res_empty["status"], "NO_HANDS_DETECTED")
        self.assertIsNone(res_empty["predicted_class"])
        self.assertFalse(res_empty["is_reliable"])

        # Test Case 2: High threshold forces rejection on ambiguous input
        rand_feat = create_synthetic_features(left_present=True, noise_level=0.5)
        res_strict = clf.predict(rand_feat, confidence_threshold=0.999)
        if res_strict["confidence"] < 0.999:
            self.assertEqual(res_strict["status"], "LOW_CONFIDENCE")
            self.assertIsNone(res_strict["predicted_class"])

    # 12. Dataset changes and stale model detection
    def test_stale_model_detection(self):
        """Verify model detects when dataset has changed since training."""
        dataset = CustomSignDataset(self.test_dataset_path)
        dataset.add_class("SIGN_1")
        dataset.add_class("SIGN_2")

        for _ in range(6):
            dataset.add_static_sample("SIGN_1", create_synthetic_features(left_present=True))
            dataset.add_static_sample("SIGN_2", create_synthetic_features(right_present=True))

        clf = CustomSignClassifier(n_estimators=10)
        clf.train(dataset, model_save_path=self.test_model_path)

        # Right after training, model is not stale
        self.assertFalse(clf.is_stale(dataset))

        # Add new sample to dataset
        dataset.add_static_sample("SIGN_1", create_synthetic_features(left_present=True))

        # Model should now be detected as stale
        self.assertTrue(clf.is_stale(dataset))

    # 13. Camera resource cleanup
    def test_camera_resource_cleanup(self):
        """Verify camera manager stops and cleans up state properly."""
        cam = CameraManager(device_index=99)  # Non-existent index
        cam.stop()
        self.assertFalse(cam.is_running)
        self.assertIsNone(cam.cap)

    # 14. Real-time recognizer integration test
    def test_real_time_recognizer_integration(self):
        """Verify RealTimeRecognizer processes frames without typing or runtime exceptions."""
        dataset = CustomSignDataset(self.test_dataset_path)
        dataset.add_class("SIGN_A")
        dataset.add_class("SIGN_B")
        for _ in range(6):
            dataset.add_static_sample("SIGN_A", create_synthetic_features(left_present=True))
            dataset.add_static_sample("SIGN_B", create_synthetic_features(right_present=True))

        clf = CustomSignClassifier(n_estimators=10)
        clf.train(dataset, model_save_path=self.test_model_path)

        extractor = DualHandLandmarkExtractor()
        recognizer = RealTimeRecognizer(
            classifier=clf,
            extractor=extractor,
            smoothing_window=5,
            confidence_threshold=0.5,
        )

        # Blank black frame
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_dict, annotated_frame = recognizer.process_frame(blank_frame, draw=True)

        self.assertIsInstance(result_dict, dict)
        self.assertIsInstance(annotated_frame, np.ndarray)
        self.assertEqual(annotated_frame.shape, (480, 640, 3))
        self.assertIn("status", result_dict)
        self.assertIn("fps", result_dict)
        extractor.close()

    # 15. Dataset hand breakdown and auto-cleaning
    def test_dataset_hand_breakdown_and_cleaning(self):
        """Verify hand distribution breakdown and auto-cleaning of noisy single-hand samples."""
        dataset = CustomSignDataset(self.test_dataset_path)
        dataset.add_class("TWO_HANDED_SIGN")

        # Add 6 two-handed samples and 2 accidental single-hand samples
        for _ in range(6):
            dataset.add_static_sample("TWO_HANDED_SIGN", create_synthetic_features(left_present=True, right_present=True))
        for _ in range(2):
            dataset.add_static_sample("TWO_HANDED_SIGN", create_synthetic_features(left_present=False, right_present=True))

        breakdown = dataset.get_hand_breakdown()
        self.assertEqual(breakdown["TWO_HANDED_SIGN"]["both"], 6)
        self.assertEqual(breakdown["TWO_HANDED_SIGN"]["right_only"], 2)

        # Run clean
        removed, msg = dataset.clean_inconsistent_samples()
        self.assertEqual(removed, 2)
        self.assertEqual(len(dataset.samples), 6)
        new_bd = dataset.get_hand_breakdown()
        self.assertEqual(new_bd["TWO_HANDED_SIGN"]["both"], 6)
        self.assertEqual(new_bd["TWO_HANDED_SIGN"]["right_only"], 0)


if __name__ == "__main__":
    unittest.main()
