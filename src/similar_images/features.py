from __future__ import annotations

import cv2
import numpy as np

from .models import ImageFeatures
from .similarity import SimilarityWeights

_HOG_DESCRIPTOR = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)
_ORB = cv2.ORB_create(nfeatures=1200)


def _resize_stretched(image: np.ndarray, size: int) -> np.ndarray:
    """Resizes to a square without maintaining aspect ratio (stretching)."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def build_features(image_bgr: np.ndarray, weights: SimilarityWeights) -> ImageFeatures:
    # 1. Base grayscale version - needed if any structural/keypoint/hash feature is used
    needs_gray = (
        weights.phash > 0
        or weights.dhash > 0
        or weights.hog > 0
        or weights.orb > 0
        or weights.ssim > 0
        or weights.edge > 0
    )
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if needs_gray else None

    # 128x128 for structural features (HOG, SSIM, Edges)
    needs_gray_128 = weights.hog > 0 or weights.ssim > 0 or weights.edge > 0
    gray_128 = _resize_stretched(gray, 128) if (needs_gray_128 and gray is not None) else None

    # 32x32 for hashing (pHash, dHash)
    needs_gray_32 = weights.phash > 0 or weights.dhash > 0
    gray_32 = _resize_stretched(gray, 32) if (needs_gray_32 and gray is not None) else None

    # 2. HSV Histogram
    hist = None
    if weights.histogram > 0:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)

    # 3. pHash
    phash = None
    if weights.phash > 0 and gray_32 is not None:
        dct = cv2.dct(np.float32(gray_32))
        dct_low = dct[:8, :8]
        median = np.median(dct_low)
        phash = (dct_low > median).astype(np.int8).flatten()

    # 4. dHash
    dhash = None
    if weights.dhash > 0 and gray_32 is not None:
        dhash_small = cv2.resize(gray_32, (9, 8), interpolation=cv2.INTER_AREA)
        dhash = (dhash_small[:, 1:] > dhash_small[:, :-1]).astype(np.int8).flatten()

    # 5. HOG
    hog = None
    if weights.hog > 0 and gray_128 is not None:
        hog = np.array(_HOG_DESCRIPTOR.compute(gray_128)).flatten().astype(np.float32)

    # 6. Edges
    edge_signature = None
    if weights.edge > 0 and gray_128 is not None:
        edge_map = cv2.Canny(gray_128, 80, 180)
        edge_signature = edge_map.flatten().astype(np.float32) / 255.0

    # 7. ORB
    orb_descriptors = None
    orb_keypoint_count = 0
    if weights.orb > 0 and gray is not None:
        orb_keypoints, orb_descriptors = _ORB.detectAndCompute(gray, None)
        orb_keypoint_count = len(orb_keypoints) if orb_keypoints is not None else 0

    return ImageFeatures(
        histogram=hist,
        phash=phash,
        dhash=dhash,
        hog=hog,
        orb_descriptors=orb_descriptors,
        orb_keypoints=orb_keypoint_count,
        gray_resized=gray_128,
        edge_signature=edge_signature,
    )
