from __future__ import annotations

from pathlib import Path

import cv2

from .features import build_features
from .models import ImageFeatures, ImageRecord, PairResult
from .similarity import SimilarityWeights, classify_score, similarity_score


def _extract_features(records: list[ImageRecord]) -> dict[Path, ImageFeatures]:
    output: dict[Path, ImageFeatures] = {}
    for rec in records:
        image = cv2.imread(str(rec.path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        output[rec.path] = build_features(image)
    return output


def compare_all(
    records: list[ImageRecord],
    similar_threshold: float,
    duplicate_threshold: float,
    weights: SimilarityWeights,
) -> tuple[list[PairResult], list[ImageRecord]]:
    features = _extract_features(records)
    loaded_records = [r for r in records if r.path in features]

    results: list[PairResult] = []
    for idx in range(len(loaded_records)):
        left = loaded_records[idx]
        for jdx in range(idx + 1, len(loaded_records)):
            right = loaded_records[jdx]
            score = similarity_score(features[left.path], features[right.path], weights=weights)
            classifier = classify_score(
                score=score,
                duplicate_threshold=duplicate_threshold,
                similar_threshold=similar_threshold,
            )
            results.append(PairResult(left=left, right=right, score=score, classifier=classifier))

    results.sort(key=lambda x: x.score, reverse=True)
    return results, loaded_records
