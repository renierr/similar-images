from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2

from .features import build_features
from .models import ImageFeatures, ImageRecord, PairResult
from .similarity import SimilarityWeights, classify_score, similarity_score


def _extract_features(
    records: list[ImageRecord],
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[Path, ImageFeatures]:
    output: dict[Path, ImageFeatures] = {}
    total = len(records)
    for index, rec in enumerate(records, start=1):
        image = cv2.imread(str(rec.path), cv2.IMREAD_COLOR)
        if image is None:
            if on_progress:
                on_progress(index, total)
            continue
        output[rec.path] = build_features(image)
        if on_progress:
            on_progress(index, total)
    return output


def compare_all(
    records: list[ImageRecord],
    similar_threshold: float,
    duplicate_threshold: float,
    weights: SimilarityWeights,
    on_feature_progress: Callable[[int, int], None] | None = None,
    on_compare_start: Callable[[int], None] | None = None,
    on_compare_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[PairResult], list[ImageRecord]]:
    features = _extract_features(records, on_progress=on_feature_progress)
    loaded_records = [r for r in records if r.path in features]
    total_pairs = (len(loaded_records) * (len(loaded_records) - 1)) // 2
    if on_compare_start:
        on_compare_start(total_pairs)

    results: list[PairResult] = []
    done_pairs = 0
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
            done_pairs += 1
            if on_compare_progress:
                on_compare_progress(done_pairs, total_pairs)

    results.sort(key=lambda x: x.score, reverse=True)
    return results, loaded_records
