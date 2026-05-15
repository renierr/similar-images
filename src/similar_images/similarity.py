from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import ImageFeatures


@dataclass(frozen=True)
class SimilarityWeights:
    histogram: float = 0.4
    phash: float = 0.2
    hog: float = 0.4
    orb: float = 0.0

    def total(self) -> float:
        return self.histogram + self.phash + self.hog + self.orb


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _orb_similarity(features_a: ImageFeatures, features_b: ImageFeatures) -> float:
    des_a = features_a.orb_descriptors
    des_b = features_b.orb_descriptors
    if des_a is None or des_b is None:
        return 0.0
    if len(des_a) < 2 or len(des_b) < 2:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(des_a, des_b, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    denom = float(min(features_a.orb_keypoints, features_b.orb_keypoints))
    if denom <= 0.0:
        return 0.0

    score = len(good_matches) / denom
    return max(0.0, min(1.0, float(score)))


def similarity_score(
    features_a: ImageFeatures,
    features_b: ImageFeatures,
    weights: SimilarityWeights,
) -> float:
    score_hist = float(cv2.compareHist(features_a.histogram, features_b.histogram, cv2.HISTCMP_CORREL))
    score_hist = max(0.0, score_hist)

    phash_distance = np.sum(features_a.phash != features_b.phash) / len(features_a.phash)
    score_phash = 1.0 - float(phash_distance)

    score_hog = _cosine_similarity(features_a.hog, features_b.hog)
    score_hog = max(0.0, score_hog)

    score_orb = _orb_similarity(features_a, features_b)

    total_weight = weights.total()
    if total_weight <= 0.0:
        raise ValueError("At least one similarity weight must be greater than zero.")

    score = (
        (weights.histogram * score_hist)
        + (weights.phash * score_phash)
        + (weights.hog * score_hog)
        + (weights.orb * score_orb)
    ) / total_weight

    return max(0.0, min(1.0, float(score)))


def classify_score(score: float, duplicate_threshold: float, similar_threshold: float) -> str:
    if score >= duplicate_threshold:
        return "duplicate"
    if score >= similar_threshold:
        return "similar"
    return "different"
