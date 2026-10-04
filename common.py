"""
ml/common.py - single source of truth for the shape-classification pipeline.

Imported by ml/train_eval.py, app.py (Gradio) and streamlit_app.py so that training and
serving produce byte-identical feature arrays.  Anything that affects the numbers a model
sees (labels, resolution, HOG parameters, preprocessing) lives HERE and nowhere else.

Exposes
-------
Constants : CLASSES, NUM_CLASSES, CLASS_TO_IDX, HOG_SIZE, HOG_PARAMS, HOG_FEATURE_DIM
Features  : to_uint8_rgb, to_gray64, extract_hog, extract_hog_batch
Serving   : validate_model, align_proba
Evaluation: CORRUPTIONS, SEVERITIES, apply_corruption   (zero-shot target-domain simulator)

IMPORTANT: the pickled SVMs were trained with exactly these HOG_PARAMS. Changing any of them
(e.g. enabling transform_sqrt) requires regenerating features and retraining both models;
validate_model() will refuse a model whose feature width no longer matches.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog

# --------------------------------------------------------------------------- #
# Labels (index == integer label stored in y_*.npy and in model.classes_)
# --------------------------------------------------------------------------- #
CLASSES: list[str] = ["Circle", "Square", "Triangle", "Star"]
NUM_CLASSES: int = len(CLASSES)
CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

# --------------------------------------------------------------------------- #
# Feature-extraction constants
# --------------------------------------------------------------------------- #
HOG_SIZE: int = 64   # every input is resized to HOG_SIZE x HOG_SIZE grayscale before HOG

HOG_PARAMS: dict = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=False,     # NOT used in training; True shifts features and lowers accuracy
    feature_vector=True,
)


def _hog_feature_dim() -> int:
    ppc, cpb = HOG_PARAMS["pixels_per_cell"], HOG_PARAMS["cells_per_block"]
    cells_y, cells_x = HOG_SIZE // ppc[0], HOG_SIZE // ppc[1]
    blocks = (cells_y - cpb[0] + 1) * (cells_x - cpb[1] + 1)
    return blocks * cpb[0] * cpb[1] * HOG_PARAMS["orientations"]


HOG_FEATURE_DIM: int = _hog_feature_dim()   # 7*7 blocks * 2*2 cells * 9 bins = 1764


# --------------------------------------------------------------------------- #
# HOG feature extraction
# --------------------------------------------------------------------------- #
def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Coerce gray / RGBA / float images to (H, W, 3) uint8."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def to_gray64(img: np.ndarray) -> np.ndarray:
    """Any image -> (64, 64) uint8 grayscale (INTER_AREA resize, ITU-R 601 luma)."""
    gray = cv2.cvtColor(to_uint8_rgb(img), cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)


def extract_hog(img: np.ndarray) -> np.ndarray:
    """Any image -> 64x64 grayscale -> HOG vector of shape (HOG_FEATURE_DIM,), float32."""
    feats = hog(to_gray64(img), **HOG_PARAMS).astype(np.float32)
    assert feats.shape == (HOG_FEATURE_DIM,), f"HOG width {feats.shape} != ({HOG_FEATURE_DIM},)"
    return feats


def extract_hog_batch(images) -> np.ndarray:
    """Iterable of images -> (N, HOG_FEATURE_DIM) float32 matrix."""
    return np.stack([extract_hog(im) for im in images])


# --------------------------------------------------------------------------- #
# Serving helpers (training/serving contract checks)
# --------------------------------------------------------------------------- #
def validate_model(model) -> None:
    """Raise ValueError if a loaded model cannot have been trained on this pipeline's features."""
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and n_in != HOG_FEATURE_DIM:
        raise ValueError(f"Model expects {n_in} features but common.py produces {HOG_FEATURE_DIM}; "
                         "feature parameters changed - retrain with ml/train_eval.py.")
    classes = getattr(model, "classes_", None)
    if classes is not None and not set(int(c) for c in classes) <= set(range(NUM_CLASSES)):
        raise ValueError(f"Model labels {list(classes)} are not a subset of 0..{NUM_CLASSES - 1}.")


def align_proba(model, proba: np.ndarray) -> np.ndarray:
    """Map predict_proba output (ordered by model.classes_) onto CLASSES order."""
    out = np.zeros(NUM_CLASSES, dtype=np.float64)
    out[np.asarray(model.classes_, dtype=int)] = np.asarray(proba, dtype=np.float64)
    return out


# --------------------------------------------------------------------------- #
# Target-domain corruptions (zero-shot evaluation).  severity in [0, 5]; params are linearly
# interpolated between the 5 discrete levels (0 = identity).
# --------------------------------------------------------------------------- #
CORRUPTIONS = ["Fog", "Sensor Noise", "Motion Blur", "Brightness Shift", "Contrast Shift"]
SEVERITIES = [1, 2, 3, 4, 5]
_LEVELS = [0, 1, 2, 3, 4, 5]


def _p(values, s):
    return float(np.interp(s, _LEVELS, values))


def _fog(img, s, rng):
    a = _p([0, 0.20, 0.35, 0.50, 0.65, 0.80], s)           # haze opacity
    h, w = img.shape[:2]
    low = cv2.resize(rng.random((6, 6)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    haze = (0.75 + 0.25 * np.clip(low, 0, 1))[..., None] * 255.0   # bright, low-frequency haze
    return (1 - a) * img + a * haze


def _sensor_noise(img, s, rng):
    return img + rng.normal(0.0, _p([0, 8, 16, 28, 42, 60], s), img.shape)


def _motion_blur(img, s, rng):
    k = int(round(_p([1, 3, 5, 9, 13, 17], s)))
    if k <= 1:
        return img
    k |= 1                                                  # odd kernel size
    angle = np.deg2rad(rng.uniform(0, 180))
    kernel = np.zeros((k, k), np.float32)
    c = k // 2
    dx, dy = np.cos(angle) * c, np.sin(angle) * c
    cv2.line(kernel, (int(round(c - dx)), int(round(c - dy))),
             (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    kernel /= kernel.sum()
    return cv2.filter2D(img.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT)


def _brightness(img, s, rng):
    return img + _p([0, 30, 60, 90, 120, 150], s)


def _contrast(img, s, rng):
    f = _p([1.0, 0.70, 0.50, 0.35, 0.20, 0.10], s)
    m = img.mean()
    return m + f * (img - m)


_FUNCS = {"Fog": _fog, "Sensor Noise": _sensor_noise, "Motion Blur": _motion_blur,
          "Brightness Shift": _brightness, "Contrast Shift": _contrast}


def apply_corruption(img: np.ndarray, name: str, severity: float, seed: int | None = None) -> np.ndarray:
    """Apply a named corruption at severity in [0, 5] and return uint8 RGB."""
    img = to_uint8_rgb(img)
    if name in (None, "None") or severity <= 0:
        return img
    if name not in _FUNCS:
        raise ValueError(f"Unknown corruption '{name}'. Choose from {CORRUPTIONS}")
    rng = np.random.default_rng(seed)
    out = _FUNCS[name](img.astype(np.float32), float(severity), rng)
    return np.clip(out, 0, 255).astype(np.uint8)
