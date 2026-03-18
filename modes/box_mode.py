"""Box detection mode – detect box-like (cuboid) objects in real-time.

Uses a classic edge-detection pipeline: grayscale → Gaussian blur →
Canny → contour finding → polygon approximation, filtered by vertex
count, area, convexity, and aspect ratio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .base import BaseMode

logger = logging.getLogger(__name__)

# Filtering thresholds
_MIN_AREA = 1000
_MAX_AREA_RATIO = 0.9  # fraction of total frame area
_MIN_ASPECT = 0.2
_MAX_ASPECT = 5.0


@dataclass
class BoxDetection:
    """A single detected box-like contour."""

    contour: np.ndarray
    bounding_rect: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]
    area: float


class BoxMode(BaseMode):
    """Real-time box / cuboid detection mode."""

    def __init__(self) -> None:
        self._detections: List[BoxDetection] = []

    @property
    def detections(self) -> List[BoxDetection]:
        """Boxes detected in the most recent frame."""
        return list(self._detections)

    def run(self, frame: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """Detect box-like shapes and annotate *frame*."""
        h, w = frame.shape[:2]
        max_area = _MAX_AREA_RATIO * h * w

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        boxes: List[BoxDetection] = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            area = cv2.contourArea(approx)
            if area < _MIN_AREA or area > max_area:
                continue
            if not cv2.isContourConvex(approx):
                continue
            x, y, bw, bh = cv2.boundingRect(approx)
            aspect = float(bw) / bh if bh > 0 else 0.0
            if aspect < _MIN_ASPECT or aspect > _MAX_ASPECT:
                continue
            cx = x + bw // 2
            cy = y + bh // 2
            boxes.append(
                BoxDetection(
                    contour=approx,
                    bounding_rect=(x, y, bw, bh),
                    center=(cx, cy),
                    area=area,
                )
            )

        self._detections = boxes

        # --- Visualisation ---
        vis = frame.copy()
        for i, box in enumerate(boxes):
            cv2.drawContours(vis, [box.contour], -1, (0, 255, 0), 2)
            x, y, bw, bh = box.bounding_rect
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (255, 0, 0), 1)
            cv2.putText(
                vis,
                f"Box {i}",
                (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )

        fps_display = context.get("fps", 0.0)
        cv2.putText(
            vis,
            f"Boxes: {len(boxes)}  FPS: {fps_display:.1f}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

        headless = context.get("headless", False)
        if headless:
            frame_idx = context.get("frame_idx", 0)
            print(f"[frame {frame_idx}] boxes_detected={len(boxes)}")

        return vis
