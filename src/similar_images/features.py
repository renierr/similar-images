from __future__ import annotations

import cv2
import numpy as np

from .models import ImageFeatures

_HOG_DESCRIPTOR = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)
_ORB = cv2.ORB_create(nfeatures=1200)


def _resize_stretched(image: np.ndarray, size: int) -> np.ndarray:
    """Resizes to a square without maintaining aspect ratio (stretching)."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def build_features(image_bgr: np.ndarray) -> ImageFeatures:
    # 1. Grayscale versions for different purposes
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # 128x128 for structural features (HOG, SSIM, Edges)
    # Note: We stretch here too to avoid padding artifacts that ruin similarity scores
    gray_128 = _resize_stretched(gray, 128)
    
    # 32x32 for hashing (pHash, dHash)
    gray_32 = _resize_stretched(gray, 32)

    # 2. HSV Histogram (more robust to quality/lighting than BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Using 8 bins for Hue and 8 for Saturation to keep it compact but descriptive
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # 3. pHash (Perceptual Hash) - Use the 32x32 version
    dct = cv2.dct(np.float32(gray_32))
    dct_low = dct[:8, :8]
    median = np.median(dct_low)
    phash = (dct_low > median).astype(np.int8).flatten()

    # 4. dHash (Difference Hash) - Use the 32x32 version
    # Compare adjacent pixels horizontally
    diff = gray_32[:, 1:] > gray_32[:, :-1]
    # Resize slightly or take a sub-region to get a standard bit length (e.g. 8x8 = 64 bits)
    # For now, let's use a 9x8 area to get 64 bits
    dhash_small = cv2.resize(gray_32, (9, 8), interpolation=cv2.INTER_AREA)
    dhash = (dhash_small[:, 1:] > dhash_small[:, :-1]).astype(np.int8).flatten()

    # 5. HOG - Use the 128x128 version
    hog = np.array(_HOG_DESCRIPTOR.compute(gray_128)).flatten().astype(np.float32)

    # 6. Edges - Use the 128x128 version
    edge_map = cv2.Canny(gray_128, 80, 180)
    edge_signature = edge_map.flatten().astype(np.float32) / 255.0

    # 7. ORB - Use original resolution for best keypoint detection
    orb_keypoints, orb_descriptors = _ORB.detectAndCompute(gray, None)
    orb_keypoint_count = len(orb_keypoints) if orb_keypoints is not None else 0

    return ImageFeatures(
        histogram=hist.astype(np.float32),
        phash=phash,
        dhash=dhash,
        hog=hog,
        orb_descriptors=orb_descriptors,
        orb_keypoints=orb_keypoint_count,
        gray_resized=gray_128,  # Keep 128 for SSIM compatibility
        edge_signature=edge_signature,
    )
