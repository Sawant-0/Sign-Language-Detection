"""
Dual-Hand Landmark Extraction and Feature Normalization Module.
Uses Google MediaPipe Hands to track both hands simultaneously (up to 21 landmarks per hand)
and outputs a unified 131-dimensional feature vector:
  - 63 features: Left hand normalized (x, y, z) coordinates (relative to left wrist and hand scale)
  - 1 feature:   Left hand presence indicator (1.0 = present, 0.0 = absent)
  - 63 features: Right hand normalized (x, y, z) coordinates (relative to right wrist and hand scale)
  - 1 feature:   Right hand presence indicator (1.0 = present, 0.0 = absent)
  - 3 features:  Inter-hand relative wrist position (dx, dy, dz) when both hands are visible
Total: 63 + 1 + 63 + 1 + 3 = 131 features.
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

FEATURE_VECTOR_DIM = 131


class DualHandLandmarkExtractor:
    """
    Extracts, normalizes, and visualizes landmarks for both hands simultaneously.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.45,
    ):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.detector = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def set_sensitivity(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.45):
        """Allows dynamically updating detection sensitivity."""
        if (
            abs(self.min_detection_confidence - min_detection_confidence) > 0.01
            or abs(self.min_tracking_confidence - min_tracking_confidence) > 0.01
        ):
            self.min_detection_confidence = min_detection_confidence
            self.min_tracking_confidence = min_tracking_confidence
            if hasattr(self, "detector") and self.detector is not None:
                self.detector.close()
            self.detector = mp_hands.Hands(
                static_image_mode=self.static_image_mode,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    def extract_features(
        self, frame: np.ndarray, draw: bool = True
    ) -> Tuple[np.ndarray, bool, bool, np.ndarray, Dict[str, Any]]:
        """
        Processes a BGR frame and extracts the 131-dimensional dual-hand feature vector.

        Returns:
            features: np.ndarray of shape (131,)
            left_detected: bool indicating whether left hand was detected
            right_detected: bool indicating whether right hand was detected
            annotated_frame: BGR frame with landmark drawings and tracking overlays
            metadata: dict containing raw pixel coordinates, bounding boxes, etc.
        """
        annotated_frame = frame.copy()
        height, width, _ = frame.shape

        # MediaPipe requires RGB image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.detector.process(rgb_frame)
        rgb_frame.flags.writeable = True

        left_landmarks = None
        right_landmarks = None
        left_raw_wrist = None
        right_raw_wrist = None

        bboxes = {"left": None, "right": None}

        # Gather all detected hands
        detected_hands = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                pred_label = handedness.classification[0].label  # 'Left' or 'Right'
                score = handedness.classification[0].score

                raw_coords = []
                px_xs = []
                px_ys = []

                for lm in hand_lms.landmark:
                    raw_coords.append([lm.x, lm.y, lm.z])
                    px_xs.append(int(lm.x * width))
                    px_ys.append(int(lm.y * height))

                raw_arr = np.array(raw_coords, dtype=np.float32)  # (21, 3)
                bbox = (
                    max(0, min(px_xs) - 15),
                    max(0, min(px_ys) - 15),
                    min(width, max(px_xs) + 15),
                    min(height, max(px_ys) + 15),
                )
                wrist_x = raw_arr[0, 0]  # normalized x coordinate of wrist

                detected_hands.append({
                    "hand_lms": hand_lms,
                    "pred_label": pred_label,
                    "score": score,
                    "raw_arr": raw_arr,
                    "bbox": bbox,
                    "wrist_x": wrist_x,
                })

        # Disambiguate and assign hands robustly without discarding second hand:
        assigned_hands = {}
        if len(detected_hands) == 1:
            h = detected_hands[0]
            assigned_hands[h["pred_label"]] = h
        elif len(detected_hands) >= 2:
            h0 = detected_hands[0]
            h1 = detected_hands[1]
            if h0["pred_label"] != h1["pred_label"]:
                # MediaPipe returned distinct labels
                assigned_hands[h0["pred_label"]] = h0
                assigned_hands[h1["pred_label"]] = h1
            else:
                # Handedness collision: MediaPipe gave the same label to both hands
                # In mirrored selfie view, user's left hand is on screen left (smaller wrist_x)
                if h0["wrist_x"] <= h1["wrist_x"]:
                    assigned_hands["Left"] = h0
                    assigned_hands["Right"] = h1
                else:
                    assigned_hands["Left"] = h1
                    assigned_hands["Right"] = h0

        # Extract Left Hand
        if "Left" in assigned_hands:
            h_left = assigned_hands["Left"]
            left_landmarks = h_left["raw_arr"]
            left_raw_wrist = h_left["raw_arr"][0].copy()
            bboxes["left"] = h_left["bbox"]
            if draw:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    h_left["hand_lms"],
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                x1, y1, x2, y2 = h_left["bbox"]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
                cv2.putText(
                    annotated_frame,
                    "Left Hand",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 120, 0),
                    2,
                )

        # Extract Right Hand
        if "Right" in assigned_hands:
            h_right = assigned_hands["Right"]
            right_landmarks = h_right["raw_arr"]
            right_raw_wrist = h_right["raw_arr"][0].copy()
            bboxes["right"] = h_right["bbox"]
            if draw:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    h_right["hand_lms"],
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                x1, y1, x2, y2 = h_right["bbox"]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(
                    annotated_frame,
                    "Right Hand",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2,
                )

        # 1. Left Hand Features (63 coords + 1 presence)
        if left_landmarks is not None:
            left_rel = left_landmarks - left_landmarks[0]
            left_scale = np.max(np.abs(left_rel))
            if left_scale < 1e-6:
                left_scale = 1.0
            left_feat = (left_rel / left_scale).flatten()  # 63
            left_presence = 1.0
            left_detected = True
        else:
            left_feat = np.zeros(63, dtype=np.float32)
            left_presence = 0.0
            left_detected = False

        # 2. Right Hand Features (63 coords + 1 presence)
        if right_landmarks is not None:
            right_rel = right_landmarks - right_landmarks[0]
            right_scale = np.max(np.abs(right_rel))
            if right_scale < 1e-6:
                right_scale = 1.0
            right_feat = (right_rel / right_scale).flatten()  # 63
            right_presence = 1.0
            right_detected = True
        else:
            right_feat = np.zeros(63, dtype=np.float32)
            right_presence = 0.0
            right_detected = False

        # 3. Inter-Hand Vector (3 coords: dx, dy, dz between wrists)
        if left_raw_wrist is not None and right_raw_wrist is not None:
            inter_hand = right_raw_wrist - left_raw_wrist  # Shape: (3,)
        else:
            inter_hand = np.zeros(3, dtype=np.float32)

        # Assemble full 131-dimensional feature vector
        features = np.concatenate([
            left_feat,                   # 63
            np.array([left_presence], dtype=np.float32),  # 1
            right_feat,                  # 63
            np.array([right_presence], dtype=np.float32), # 1
            inter_hand                   # 3
        ]).astype(np.float32)

        metadata = {
            "left_detected": left_detected,
            "right_detected": right_detected,
            "bboxes": bboxes,
            "both_hands_detected": left_detected and right_detected,
            "any_hand_detected": left_detected or right_detected,
        }

        return features, left_detected, right_detected, annotated_frame, metadata

    def close(self):
        """Cleanly releases MediaPipe resources."""
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()


def create_synthetic_features(
    left_present: bool = True,
    right_present: bool = False,
    noise_level: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generates a 131-dimensional feature vector strictly for software unit tests.
    """
    if seed is not None:
        np.random.seed(seed)

    features = np.zeros(FEATURE_VECTOR_DIM, dtype=np.float32)

    if left_present:
        left_vals = np.random.uniform(-1.0, 1.0, size=63).astype(np.float32)
        features[0:63] = left_vals / (np.max(np.abs(left_vals)) + 1e-6)
        features[63] = 1.0  # Left presence

    if right_present:
        right_vals = np.random.uniform(-1.0, 1.0, size=63).astype(np.float32)
        features[64:127] = right_vals / (np.max(np.abs(right_vals)) + 1e-6)
        features[127] = 1.0  # Right presence

    if left_present and right_present:
        features[128:131] = np.random.uniform(-0.5, 0.5, size=3).astype(np.float32)

    if noise_level > 0.0:
        noise = np.random.normal(0, noise_level, size=FEATURE_VECTOR_DIM).astype(np.float32)
        if left_present:
            features[0:63] += noise[0:63]
        if right_present:
            features[64:127] += noise[64:127]
        if left_present and right_present:
            features[128:131] += noise[128:131]

    return features
