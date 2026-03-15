"""robo-eye-sense – lightweight real-time visual marker detection.

Primary public surface
----------------------
* :class:`~robo_eye_sense.detector.RoboEyeDetector` – all-in-one detector/tracker
* :class:`~robo_eye_sense.results.Detection` – per-detection data class
* :class:`~robo_eye_sense.results.DetectionType` – detection category enum
* :class:`~robo_eye_sense.camera.Camera` – camera capture helper
"""

from .results import Detection, DetectionType

__all__ = ["RoboEyeDetector", "Detection", "DetectionType"]
__version__ = "0.1.0"


def __getattr__(name: str):
    """Expose heavy imports lazily to keep lightweight modules usable.

    Importing :mod:`robo_eye_sense` should not require OpenCV unless the
    caller actually accesses :class:`RoboEyeDetector`.
    """

    if name == "RoboEyeDetector":
        from .detector import RoboEyeDetector

        return RoboEyeDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
