"""Pose estimation mode – estimate 6-DoF pose of AprilTags.

Loads camera calibration from a ``.npz`` file, detects AprilTags using
``pupil_apriltags``, and estimates their pose with ``cv2.solvePnP``.
Visualises tag borders, IDs, and 3-D axes on the frame.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseMode

logger = logging.getLogger(__name__)


def _load_calibration(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera matrix and distortion coefficients from *path*.

    Returns
    -------
    (camera_matrix, dist_coeffs)

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If required keys are missing from the file.
    """
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"]


def _get_tag_corners_3d(tag_size: float) -> np.ndarray:
    """Return the four 3-D corners of a tag centred at origin."""
    half = tag_size / 2.0
    return np.array([
        [-half, -half, 0],
        [ half, -half, 0],
        [ half,  half, 0],
        [-half,  half, 0],
    ], dtype=np.float64)


class PoseMode(BaseMode):
    """AprilTag 6-DoF pose estimation mode.

    Parameters
    ----------
    tag_size:
        Physical size (side length) of the tags in metres.
    calibration_path:
        Path to the calibration ``.npz`` file.  When ``None`` a default
        camera matrix is synthesised from the frame dimensions.
    """

    def __init__(
        self,
        tag_size: float = 0.05,
        calibration_path: Optional[str] = None,
    ) -> None:
        self._tag_size = tag_size
        self._obj_pts = _get_tag_corners_3d(tag_size)
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None

        if calibration_path is not None:
            try:
                self._camera_matrix, self._dist_coeffs = _load_calibration(
                    calibration_path,
                )
                logger.info("Loaded calibration from %s", calibration_path)
            except (FileNotFoundError, KeyError) as exc:
                logger.warning(
                    "Could not load calibration from %s: %s – "
                    "falling back to default matrix.",
                    calibration_path, exc,
                )

        # Lazy-load the AprilTag detector
        self._detector: Any = None

    # ------------------------------------------------------------------
    def _ensure_detector(self) -> None:
        """Create the AprilTag detector on first use."""
        if self._detector is not None:
            return
        try:
            from pupil_apriltags import Detector  # type: ignore[import-untyped]

            self._detector = Detector(families="tag36h11")
            logger.info("AprilTag detector initialised (tag36h11)")
        except ImportError:
            try:
                import apriltag  # type: ignore[import-untyped]

                self._detector = apriltag.Detector()
                logger.info("AprilTag detector initialised (apriltag fallback)")
            except ImportError:
                logger.error(
                    "Neither pupil_apriltags nor apriltag is installed."
                )
                self._detector = None

    # ------------------------------------------------------------------
    def _default_camera_matrix(self, w: int, h: int) -> np.ndarray:
        """Synthesise a camera matrix assuming ~60° HFOV."""
        fx = w / (2.0 * np.tan(np.radians(30)))
        fy = fx
        return np.array([
            [fx, 0, w / 2.0],
            [0, fy, h / 2.0],
            [0, 0, 1],
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    def run(self, frame: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """Detect AprilTags and estimate their poses."""
        self._ensure_detector()
        if self._detector is None:
            return frame

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cam_mtx = (
            self._camera_matrix
            if self._camera_matrix is not None
            else self._default_camera_matrix(w, h)
        )
        dist = (
            self._dist_coeffs
            if self._dist_coeffs is not None
            else np.zeros(5, dtype=np.float64)
        )

        # Detect tags
        try:
            tags = self._detector.detect(gray)
        except Exception:
            logger.exception("AprilTag detection failed")
            return frame

        vis = frame.copy()

        for tag in tags:
            # Extract corners (different attribute names per library)
            corners = getattr(tag, "corners", None)
            if corners is None:
                continue
            corners_2d = np.array(corners, dtype=np.float64).reshape(-1, 2)
            tag_id = getattr(tag, "tag_id", "?")

            # Draw tag border
            pts = corners_2d.astype(int).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

            # Draw tag ID
            cx = int(corners_2d[:, 0].mean())
            cy = int(corners_2d[:, 1].mean())
            cv2.putText(
                vis, f"ID:{tag_id}", (cx - 20, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
            )

            # Pose estimation
            success, rvec, tvec = cv2.solvePnP(
                self._obj_pts, corners_2d, cam_mtx, dist,
            )
            if success:
                cv2.drawFrameAxes(vis, cam_mtx, dist, rvec, tvec, self._tag_size * 0.5)
                headless = context.get("headless", False)
                if headless:
                    frame_idx = context.get("frame_idx", 0)
                    tv = tvec.flatten()
                    rv = rvec.flatten()
                    print(
                        f"[frame {frame_idx}] "
                        f"ID: {tag_id} | "
                        f"tvec: [{tv[0]:.4f}, {tv[1]:.4f}, {tv[2]:.4f}] | "
                        f"rvec: [{rv[0]:.4f}, {rv[1]:.4f}, {rv[2]:.4f}]"
                    )

        return vis
