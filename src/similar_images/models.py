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
    histogram: np.ndarray
    phash: np.ndarray
    dhash: np.ndarray
    hog: np.ndarray
    orb_descriptors: np.ndarray | None
    orb_keypoints: int
    gray_resized: np.ndarray
    edge_signature: np.ndarray


@dataclass(frozen=True)
class PairResult:
    left: ImageRecord
    right: ImageRecord
    score: float
    classifier: str
