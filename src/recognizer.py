"""
Real-Time Recognition Orchestrator Module.
Applies temporal smoothing, rolling window majority voting,
debounce event logging, and generates professional on-frame HUD visual overlays.
"""

from typing import Optional, List, Dict, Any, Deque, Tuple
from collections import deque, Counter
import time
import numpy as np
import cv2

from src.landmarks import DualHandLandmarkExtractor
from src.model import CustomSignClassifier, extract_temporal_features


class RealTimeRecognizer:
    """
    Manages real-time sign recognition pipeline, smoothing predictions over a rolling window,
    avoiding flickering, logging stable recognition events, and rendering visual HUDs.
    """

    def __init__(
        self,
        classifier: CustomSignClassifier,
        extractor: DualHandLandmarkExtractor,
        smoothing_window: int = 7,
        confidence_threshold: float = 0.65,
        dynamic_sequence_length: int = 25,
    ):
        self.classifier = classifier
        self.extractor = extractor
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.dynamic_sequence_length = dynamic_sequence_length

        # Rolling buffers
        self.prediction_buffer: Deque[Optional[str]] = deque(maxlen=smoothing_window)
        self.confidence_buffer: Deque[float] = deque(maxlen=smoothing_window)
        self.dynamic_frame_buffer: Deque[np.ndarray] = deque(maxlen=dynamic_sequence_length)

        # Event tracking
        self.last_stable_prediction: Optional[str] = None
        self.last_event_time: float = 0.0
        self.history: List[Dict[str, Any]] = []

        # FPS calculation
        self.prev_frame_time: float = time.time()
        self.fps: float = 0.0

    def process_frame(
        self, frame: np.ndarray, draw: bool = True
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Executes one complete inference cycle on a single video frame.

        Returns:
            result_dict: Metadata including stable predicted class, confidence, status, etc.
            annotated_frame: Frame with MediaPipe skeleton and visual HUD overlay.
        """
        current_time = time.time()
        fps_delta = current_time - self.prev_frame_time
        if fps_delta > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / fps_delta)
        self.prev_frame_time = current_time

        # 1. Extract 131-dimensional feature vector
        features, left_det, right_det, annotated_frame, meta = self.extractor.extract_features(
            frame, draw=draw
        )

        # 2. Check if hands are present
        if not (left_det or right_det):
            self.prediction_buffer.append(None)
            self.confidence_buffer.append(0.0)
            status = "NO_HANDS_DETECTED"
            raw_pred = None
            raw_conf = 0.0
            probabilities = {}
        else:
            # 3. Model inference based on classifier mode
            if self.classifier.mode == "dynamic":
                self.dynamic_frame_buffer.append(features)
                if len(self.dynamic_frame_buffer) >= self.dynamic_sequence_length:
                    temporal_feat = extract_temporal_features(list(self.dynamic_frame_buffer))
                    pred_res = self.classifier.predict(
                        temporal_feat, confidence_threshold=self.confidence_threshold
                    )
                else:
                    pred_res = {
                        "predicted_class": None,
                        "confidence": 0.0,
                        "probabilities": {},
                        "status": "BUFFERING_SEQUENCE",
                        "is_reliable": False,
                    }
            else:
                # Static pose classification
                pred_res = self.classifier.predict(
                    features, confidence_threshold=self.confidence_threshold
                )

            raw_pred = pred_res.get("predicted_class")
            raw_conf = pred_res.get("confidence", 0.0)
            status = pred_res.get("status", "UNKNOWN")
            probabilities = pred_res.get("probabilities", {})

            self.prediction_buffer.append(raw_pred)
            self.confidence_buffer.append(raw_conf)

        # 4. Temporal Smoothing via Majority Voting
        valid_predictions = [p for p in self.prediction_buffer if p is not None]
        if valid_predictions:
            counts = Counter(valid_predictions)
            most_common, count = counts.most_common(1)[0]
            # Must appear in at least 50% of the recent valid frames
            if count >= len(self.prediction_buffer) * 0.45:
                stable_pred = most_common
                avg_conf = float(np.mean([c for c in self.confidence_buffer if c > 0]))
            else:
                stable_pred = None
                avg_conf = float(np.mean(self.confidence_buffer)) if self.confidence_buffer else 0.0
        else:
            stable_pred = None
            avg_conf = 0.0

        # 5. Debounced History Event Logging
        if stable_pred is not None and (stable_pred != self.last_stable_prediction or (current_time - self.last_event_time) > 2.5):
            self.last_stable_prediction = stable_pred
            self.last_event_time = current_time
            self.history.append({
                "sign": stable_pred,
                "confidence": round(avg_conf, 3),
                "timestamp": current_time,
                "time_str": time.strftime("%H:%M:%S", time.localtime(current_time)),
            })

        # 6. Render On-Frame Professional HUD Overlay
        if draw:
            self._draw_hud(
                annotated_frame,
                stable_pred=stable_pred,
                confidence=avg_conf,
                status=status,
                left_det=left_det,
                right_det=right_det,
            )

        result = {
            "stable_prediction": stable_pred,
            "raw_prediction": raw_pred,
            "confidence": round(avg_conf, 3),
            "status": status,
            "left_detected": left_det,
            "right_detected": right_det,
            "fps": round(self.fps, 1),
            "probabilities": probabilities,
            "history_count": len(self.history),
        }
        return result, annotated_frame

    def _draw_hud(
        self,
        frame: np.ndarray,
        stable_pred: Optional[str],
        confidence: float,
        status: str,
        left_det: bool,
        right_det: bool,
    ):
        """
        Draws semi-transparent HUD banner, prediction pill, and tracking badges.
        """
        h, w, _ = frame.shape

        # Top Banner Background (dark translucent bar)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 75), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Hand Status Badges
        left_color = (0, 220, 100) if left_det else (120, 120, 120)
        right_color = (0, 220, 100) if right_det else (120, 120, 120)

        cv2.circle(frame, (25, 25), 6, left_color, -1)
        cv2.putText(frame, "L Hand", (38, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)

        cv2.circle(frame, (25, 50), 6, right_color, -1)
        cv2.putText(frame, "R Hand", (38, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)

        # Main Prediction Display (Center)
        if stable_pred:
            display_text = f"SIGN: {stable_pred}"
            conf_pct = int(confidence * 100)
            conf_text = f"Confidence: {conf_pct}%"
            text_color = (0, 255, 127)  # Bright Green
        elif status == "LOW_CONFIDENCE":
            display_text = "Low Confidence (Unrecognized)"
            conf_text = f"Below Threshold ({int(self.confidence_threshold * 100)}%)"
            text_color = (0, 190, 255)  # Orange
        elif status == "NO_HANDS_DETECTED":
            display_text = "No Hands in Frame"
            conf_text = "Raise hand to start recognition"
            text_color = (180, 180, 180)  # Gray
        else:
            display_text = "Analyzing Gesture..."
            conf_text = "Keep hand in steady view"
            text_color = (255, 255, 255)

        cv2.putText(frame, display_text, (130, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)
        cv2.putText(frame, conf_text, (130, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        # FPS Counter (Top Right)
        fps_text = f"{int(self.fps)} FPS"
        cv2.putText(frame, fps_text, (w - 85, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    def clear_history(self):
        """Clears logged recognition events."""
        self.history = []
        self.last_stable_prediction = None
