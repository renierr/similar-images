from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import ImageFeatures


@dataclass(frozen=True)
class SimilarityWeights:
    histogram: float = 0.3
    phash: float = 0.2
    dhash: float = 0.2
    hog: float = 0.3
    orb: float = 0.0
    ssim: float = 0.0
    edge: float = 0.0

    def total(self) -> float:
        return (
            self.histogram
            + self.phash
            + self.dhash
            + self.hog
            + self.orb
            + self.ssim
            + self.edge
        )


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


def _ssim_similarity(features_a: ImageFeatures, features_b: ImageFeatures) -> float:
    gray_a = features_a.gray_resized.astype(np.float32)
    gray_b = features_b.gray_resized.astype(np.float32)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(gray_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gray_b, (11, 11), 1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(gray_a * gray_a, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(gray_b * gray_b, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(gray_a * gray_b, (11, 11), 1.5) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    denominator = np.where(denominator == 0, 1e-8, denominator)

    ssim_map = numerator / denominator
    ssim_value = float(np.mean(ssim_map))
    return max(0.0, min(1.0, ssim_value))


def _edge_similarity(features_a: ImageFeatures, features_b: ImageFeatures) -> float:
    return max(0.0, min(1.0, _cosine_similarity(features_a.edge_signature, features_b.edge_signature)))


def similarity_score(
    features_a: ImageFeatures,
    features_b: ImageFeatures,
    weights: SimilarityWeights,
) -> float:
    score_hist = float(cv2.compareHist(features_a.histogram, features_b.histogram, cv2.HISTCMP_CORREL))
    score_hist = max(0.0, score_hist)

    phash_distance = np.sum(features_a.phash != features_b.phash) / len(features_a.phash)
    score_phash = 1.0 - float(phash_distance)

    dhash_distance = np.sum(features_a.dhash != features_b.dhash) / len(features_a.dhash)
    score_dhash = 1.0 - float(dhash_distance)

    score_hog = _cosine_similarity(features_a.hog, features_b.hog)
    score_hog = max(0.0, score_hog)

    score_orb = _orb_similarity(features_a, features_b)
    score_ssim = _ssim_similarity(features_a, features_b)
    score_edge = _edge_similarity(features_a, features_b)

    total_weight = weights.total()
    if total_weight <= 0.0:
        raise ValueError("At least one similarity weight must be greater than zero.")

    score = (
        (weights.histogram * score_hist)
        + (weights.phash * score_phash)
        + (weights.dhash * score_dhash)
        + (weights.hog * score_hog)
        + (weights.orb * score_orb)
        + (weights.ssim * score_ssim)
        + (weights.edge * score_edge)
    ) / total_weight

    return max(0.0, min(1.0, float(score)))


def classify_score(score: float, duplicate_threshold: float, similar_threshold: float) -> str:
    if score >= duplicate_threshold:
        return "duplicate"
    if score >= similar_threshold:
        return "similar"
    return "different"
