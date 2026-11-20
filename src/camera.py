"""
Webcam Management Module for Real-Time Frame Capture.
Provides resilient device initialization, frame acquisition, resolution configuration,
and clean hardware resource deallocation on Windows.
"""

from typing import Optional, Tuple
import cv2
import time


class CameraManager:
    """
    Manages OpenCV VideoCapture with fallback device indices and graceful error handling.
    """

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self.preferred_index = device_index
        self.target_width = width
        self.target_height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self.active_index: Optional[int] = None

    def start(self) -> Tuple[bool, str]:
        """
        Initializes and opens the camera device. Tries preferred index first,
        then falls back to secondary indices (0, 1, 2) if necessary.
        """
        if self.is_running and self.cap is not None and self.cap.isOpened():
            return True, f"Camera already active on index {self.active_index}."

        candidate_indices = [self.preferred_index]
        for idx in [0, 1, 2]:
            if idx not in candidate_indices:
                candidate_indices.append(idx)

        last_error = ""
        for idx in candidate_indices:
            try:
                # Use DirectShow backend on Windows (cv2.CAP_DSHOW) for faster startup and cleaner release
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    # Fallback to default backend
                    cap.release()
                    cap = cv2.VideoCapture(idx)

                if cap.isOpened():
                    # Verify that we can actually read a valid frame
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                        cap.set(cv2.CAP_PROP_FPS, 30)

                        self.cap = cap
                        self.active_index = idx
                        self.is_running = True
                        return True, f"Camera successfully connected on device index {idx}."
                    else:
                        cap.release()
                        last_error = f"Device index {idx} opened but failed to yield video frames."
                else:
                    last_error = f"Device index {idx} could not be opened."
            except Exception as e:
                last_error = str(e)

        self.cap = None
        self.is_running = False
        return (
            False,
            f"Unable to access camera device. Last error: {last_error}. "
            "Please verify that your webcam is connected and not in use by another application.",
        )

    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Reads a single BGR video frame.
        Returns: (success: bool, frame: Optional[np.ndarray])
        """
        if not self.is_running or self.cap is None:
            return False, None

        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Flip horizontally for natural mirror/selfie view
                flipped = cv2.flip(frame, 1)
                return True, flipped
            return False, None
        except Exception:
            return False, None

    def stop(self) -> None:
        """
        Releases the webcam hardware and cleans up OpenCV capture resources.
        """
        self.is_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                print(f"Warning during camera release: {e}")
            finally:
                self.cap = None
                self.active_index = None

    def __del__(self):
        self.stop()
