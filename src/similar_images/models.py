from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    name: str


@dataclass(frozen=True)
class ImageFeatures:
    histogram: np.ndarray | None
    phash: np.ndarray | None
    dhash: np.ndarray | None
    hog: np.ndarray | None
    orb_descriptors: np.ndarray | None
    orb_keypoints: int
    gray_resized: np.ndarray | None
    edge_signature: np.ndarray | None


@dataclass(frozen=True)
class PairResult:
    left: ImageRecord
    right: ImageRecord
    score: float
    classifier: str
