from __future__ import annotations

import cv2
import numpy as np

from .models import ImageFeatures

_HOG_DESCRIPTOR = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)
_ORB = cv2.ORB_create(nfeatures=1200)


def _resize_and_pad(image: np.ndarray, target_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Image with zero width or height cannot be processed.")

    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))

    resized = cv2.resize(image, (new_w, new_h))
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def build_features(image_bgr: np.ndarray) -> ImageFeatures:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_resized = _resize_and_pad(gray, 128)

    hist = cv2.calcHist([image_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    dct = cv2.dct(np.float32(gray_resized))
    dct_low = dct[:8, :8]
    median = np.median(dct_low)
    phash = (dct_low > median).astype(np.int8).flatten()

    hog = np.array(_HOG_DESCRIPTOR.compute(gray_resized)).flatten().astype(np.float32)

    edge_map = cv2.Canny(gray_resized, 80, 180)
    edge_signature = edge_map.flatten().astype(np.float32) / 255.0

    orb_keypoints, orb_descriptors = _ORB.detectAndCompute(gray, None)
    orb_keypoint_count = len(orb_keypoints) if orb_keypoints is not None else 0

    return ImageFeatures(
        histogram=hist.astype(np.float32),
        phash=phash,
        hog=hog,
        orb_descriptors=orb_descriptors,
        orb_keypoints=orb_keypoint_count,
        gray_resized=gray_resized,
        edge_signature=edge_signature,
    )
