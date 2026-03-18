"""Calibration mode – compute camera intrinsics from chessboard images.

Detects a chessboard pattern in live frames, lets the user capture 15–25
valid views (press SPACE), then runs ``cv2.calibrateCamera`` and saves
the result to an ``.npz`` file.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .base import BaseMode

logger = logging.getLogger(__name__)

# Sub-pixel corner refinement criteria
_TERM_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

_MIN_CAPTURES = 15
_MAX_CAPTURES = 25


class CalibrationMode(BaseMode):
    """Camera intrinsic calibration via chessboard detection.

    Parameters
    ----------
    chessboard_size:
        Number of *inner* corners as ``(cols, rows)``, e.g. ``(9, 6)``.
    output_path:
        File path for the saved calibration (``.npz``).
    """

    def __init__(
        self,
        chessboard_size: Tuple[int, int] = (9, 6),
        output_path: str = "calibration.npz",
    ) -> None:
        self._board_size = chessboard_size
        self._output_path = output_path

        # Prepare object points for one board view
        self._objp = np.zeros(
            (chessboard_size[0] * chessboard_size[1], 3), dtype=np.float32,
        )
        self._objp[:, :2] = np.mgrid[
            0:chessboard_size[0], 0:chessboard_size[1]
        ].T.reshape(-1, 2)

        self._obj_points: List[np.ndarray] = []
        self._img_points: List[np.ndarray] = []
        self._calibrated = False

    # ------------------------------------------------------------------
    @property
    def capture_count(self) -> int:
        """Number of valid chessboard captures so far."""
        return len(self._obj_points)

    @property
    def is_calibrated(self) -> bool:
        """Whether calibration has been completed."""
        return self._calibrated

    # ------------------------------------------------------------------
    def run(self, frame: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """Detect chessboard, handle SPACE capture, trigger calibration."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self._board_size, None)

        vis = frame.copy()

        if found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), _TERM_CRITERIA,
            )
            cv2.drawChessboardCorners(vis, self._board_size, corners_refined, found)

            key = context.get("key", -1)
            if key == ord(" ") and self.capture_count < _MAX_CAPTURES:
                self._obj_points.append(self._objp)
                self._img_points.append(corners_refined)
                logger.info(
                    "Captured frame %d/%d", self.capture_count, _MAX_CAPTURES,
                )

        # HUD overlay
        status = (
            f"Captures: {self.capture_count}/{_MAX_CAPTURES}  "
            f"{'[BOARD DETECTED]' if found else '[NO BOARD]'}"
        )
        cv2.putText(
            vis, status, (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
        )
        if self.capture_count >= _MIN_CAPTURES:
            cv2.putText(
                vis,
                "Press 'c' to calibrate",
                (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA,
            )

        # Trigger calibration on 'c' key
        key = context.get("key", -1)
        if (
            key == ord("c")
            and self.capture_count >= _MIN_CAPTURES
            and not self._calibrated
        ):
            self._run_calibration(gray.shape[::-1])

        # Print to stdout in headless mode
        headless = context.get("headless", False)
        if headless:
            frame_idx = context.get("frame_idx", 0)
            print(
                f"[frame {frame_idx}] calibration: "
                f"captures={self.capture_count} board_found={found}"
            )

        return vis

    # ------------------------------------------------------------------
    def _run_calibration(self, image_size: Tuple[int, int]) -> None:
        """Run ``cv2.calibrateCamera`` and save the result."""
        logger.info("Running calibration with %d captures…", self.capture_count)
        ret, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
            self._obj_points, self._img_points, image_size, None, None,
        )
        np.savez(
            self._output_path,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        self._calibrated = True
        logger.info("Calibration saved to %s (RMS=%.4f)", self._output_path, ret)
        print(
            f"Calibration complete (RMS={ret:.4f}). "
            f"Saved to {self._output_path}"
        )
