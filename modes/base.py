"""Base class for all operational modes."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class BaseMode:
    """Common interface for operational modes.

    Sub-classes must override :meth:`run` to process a single frame.
    The *context* dictionary carries per-invocation metadata such as
    ``headless`` flag, ``key`` press code, etc.
    """

    def run(self, frame: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """Process *frame* and return the annotated result.

        Parameters
        ----------
        frame:
            The current BGR video frame.
        context:
            Runtime metadata (e.g. ``headless``, ``key``, ``frame_idx``).

        Returns
        -------
        np.ndarray
            The (possibly annotated) frame for display / recording.
        """
        return frame
