"""
Standalone Desktop Application for Real-Time Sign Language Recognition.
Provides high-FPS OpenCV GUI window with dual-hand tracking overlays,
confidence estimation, and keyboard controls.

Controls:
  - 'q': Quit application
  - 'c': Clear recognition history
  - 's': Toggle pause / resume recognition
"""

import sys
import time
import cv2
import numpy as np

from src.landmarks import DualHandLandmarkExtractor
from src.dataset import CustomSignDataset, DEFAULT_DATASET_FILE
from src.model import CustomSignClassifier, DEFAULT_MODEL_PATH
from src.camera import CameraManager
from src.recognizer import RealTimeRecognizer


def main():
    print("=" * 65)
    print("  SignSense: Standalone Desktop Sign Language Detector")
    print("=" * 65)

    # 1. Check dataset and trained model
    dataset = CustomSignDataset(DEFAULT_DATASET_FILE)
    model = CustomSignClassifier.load(DEFAULT_MODEL_PATH)

    if model is None:
        print("\n[ERROR] No trained model found at 'models/trained_model.joblib'!")
        print("Please launch the web application ('streamlit run app.py') first to")
        print("record custom sign samples and train your model before using the desktop app.")
        print("=" * 65)
        sys.exit(1)

    if model.is_stale(dataset):
        print("\n[WARNING] Dataset has been modified since last model training.")
        print("Using the existing model. (Retraining via app.py recommended)\n")

    print(f"Loaded Model Mode: {model.mode.upper()}")
    print(f"Recognized Sign Classes: {', '.join(model.classes)}")
    print("\nStarting Camera (Press 'q' in video window to exit)...")

    # 2. Initialize Camera & Extractor
    camera = CameraManager(device_index=0, width=640, height=480)
    cam_ok, cam_msg = camera.start()
    if not cam_ok:
        print(f"\n[ERROR] {cam_msg}")
        sys.exit(1)

    extractor = DualHandLandmarkExtractor(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    recognizer = RealTimeRecognizer(
        classifier=model,
        extractor=extractor,
        smoothing_window=7,
        confidence_threshold=0.65,
    )

    window_name = "SignSense - Real-Time Sign Detection (Press 'q' to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    is_paused = False

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if not is_paused:
                res, annotated_frame = recognizer.process_frame(frame, draw=True)
            else:
                annotated_frame = frame
                cv2.putText(
                    annotated_frame,
                    "PAUSED (Press 's' to resume)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            # Display window
            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                recognizer.clear_history()
                print("Recognition history cleared.")
            elif key == ord("s"):
                is_paused = not is_paused
                print("Recognition paused." if is_paused else "Recognition resumed.")

    finally:
        camera.stop()
        extractor.close()
        cv2.destroyAllWindows()
        print("\nCamera released and window closed cleanly. Exiting.")


if __name__ == "__main__":
    main()
