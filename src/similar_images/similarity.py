from __future__ import annotations

import cv2
import numpy as np

from .models import ImageFeatures


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def similarity_score(features_a: ImageFeatures, features_b: ImageFeatures) -> float:
    score_hist = float(cv2.compareHist(features_a.histogram, features_b.histogram, cv2.HISTCMP_CORREL))
    score_hist = max(0.0, score_hist)

    phash_distance = np.sum(features_a.phash != features_b.phash) / len(features_a.phash)
    score_phash = 1.0 - float(phash_distance)

    score_hog = _cosine_similarity(features_a.hog, features_b.hog)
    score_hog = max(0.0, score_hog)

    score = (0.4 * score_hist) + (0.2 * score_phash) + (0.4 * score_hog)
    return max(0.0, min(1.0, float(score)))


def classify_score(score: float, duplicate_threshold: float, similar_threshold: float) -> str:
    if score >= duplicate_threshold:
        return "duplicate"
    if score >= similar_threshold:
        return "similar"
    return "different"
