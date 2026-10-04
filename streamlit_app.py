"""
Streamlit dashboard - Out-of-Distribution (OOD) Domain Adaptation study.

Compares a Naive Baseline SVM (trained on clean shapes) with a Domain-Randomized SVM
(trained on randomized shapes) on the SAME user-supplied, synthetically corrupted image.

Run:  streamlit run streamlit_app.py

Self-contained on purpose (no imports from ml/), so it can be deployed on its own with
just this file, the two .pkl files and requirements.txt.  The feature pipeline below
mirrors ml/common.py exactly (see HOG_PARAMS).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from skimage.feature import hog

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {"baseline": ROOT / "baseline_model.pkl", "randomized": ROOT / "randomized_model.pkl"}
CLASSES = ["Circle", "Square", "Triangle", "Star"]          # label order used in training (0..3)
CORRUPTIONS = ["Sensor Noise", "Motion Blur", "Fog / Haze"]
TRAIN_RES = 96     # resolution the corruptions were calibrated at (training images are 96x96)
HOG_SIZE = 64
NOISE_SEED = 42    # fixed so Streamlit reruns (every widget change) don't make the noise flicker

# HOG parameters.  These MUST equal the ones used to train the pickled models (ml/common.py).
# transform_sqrt was NOT used in training; enabling it shifts the features and lowers accuracy
# (held-out check: randomized model 90.7% -> 89.0%), so it is exposed as an explicit switch.
HOG_TRANSFORM_SQRT = False
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                  block_norm="L2-Hys", transform_sqrt=HOG_TRANSFORM_SQRT, feature_vector=True)

# Aggregate figures as specified for the paper.  Edit here if the paper's numbers change.
PAPER_STATS = {"Naive Baseline": 25.3, "Domain Randomized": 57.1}
BAR_COLORS = {"baseline": "#c0392b", "randomized": "#1f6feb"}


# --------------------------------------------------------------------------- #
# 1. Model initialisation (cached, never raises)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading SVM models...")
def load_models() -> tuple[dict, dict]:
    """Load both pickles once per server process. Returns (models, errors)."""
    models, errors = {}, {}
    for key, path in MODEL_PATHS.items():
        try:
            models[key] = joblib.load(path)
        except Exception as exc:  # missing file, sklearn version mismatch, corrupt pickle...
            errors[key] = f"{type(exc).__name__}: {exc}"
    return models, errors


# --------------------------------------------------------------------------- #
# 2. Feature extraction
# --------------------------------------------------------------------------- #
def extract_hog(img_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 image -> 64x64 grayscale -> HOG vector (1764-d), identical to training."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)
    return hog(gray, **HOG_PARAMS).astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. Target-domain simulator (severity 0..5, continuous; 0 = identity)
# --------------------------------------------------------------------------- #
_LEVELS = [0, 1, 2, 3, 4, 5]


def _p(values, s: float) -> float:
    return float(np.interp(s, _LEVELS, values))


def apply_corruption(img_rgb: np.ndarray, name: str, severity: float) -> np.ndarray:
    """Apply Sensor Noise / Motion Blur / Fog-Haze at the given severity (same parameters as training eval)."""
    if severity <= 0:
        return img_rgb
    rng = np.random.default_rng(NOISE_SEED)
    x = img_rgb.astype(np.float32)
    h, w = x.shape[:2]

    if name == "Sensor Noise":
        x = x + rng.normal(0.0, _p([0, 8, 16, 28, 42, 60], severity), x.shape)

    elif name == "Motion Blur":
        k = int(round(_p([1, 3, 5, 9, 13, 17], severity))) | 1       # odd kernel length
        if k > 1:
            angle = np.deg2rad(rng.uniform(0, 180))
            c = k // 2
            kernel = np.zeros((k, k), np.float32)
            cv2.line(kernel,
                     (int(round(c - np.cos(angle) * c)), int(round(c - np.sin(angle) * c))),
                     (int(round(c + np.cos(angle) * c)), int(round(c + np.sin(angle) * c))), 1.0, 1)
            kernel /= kernel.sum()
            x = cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REFLECT)

    elif name == "Fog / Haze":
        a = _p([0, 0.20, 0.35, 0.50, 0.65, 0.80], severity)          # haze opacity
        low = cv2.resize(rng.random((6, 6)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        haze = (0.75 + 0.25 * np.clip(low, 0, 1))[..., None] * 255.0
        x = (1 - a) * x + a * haze

    else:
        raise ValueError(f"Unknown corruption: {name}")
    return np.clip(x, 0, 255).astype(np.uint8)


def show_image(container, img: np.ndarray, caption: str):
    """Full-width image across Streamlit versions (width='stretch' is new; use_container_width is legacy)."""
    try:
        container.image(img, caption=caption, width="stretch")
    except Exception:
        container.image(img, caption=caption, use_container_width=True)


def load_uploaded_image(uploaded) -> np.ndarray:
    """Decode an upload to a square RGB uint8 array at training resolution (aspect-preserving pad)."""
    img = Image.open(uploaded)
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    arr = cv2.copyMakeBorder(arr, top, side - h - top, left, side - w - left, cv2.BORDER_REPLICATE)
    return cv2.resize(arr, (TRAIN_RES, TRAIN_RES), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# Inference (fail-safe)
# --------------------------------------------------------------------------- #
def predict_confidences(model, img_rgb: np.ndarray) -> pd.DataFrame:
    """Return a one-column DataFrame (index = CLASSES) of class probabilities. Raises on failure."""
    proba = model.predict_proba(extract_hog(img_rgb)[None, :])[0]
    out = np.zeros(len(CLASSES))
    out[np.asarray(model.classes_, dtype=int)] = proba
    return pd.DataFrame({"Confidence": out}, index=pd.Index(CLASSES, name="Class"))


def render_prediction_panel(title: str, key: str, models: dict, errors: dict, img_rgb: np.ndarray | None):
    """One model's result card. Every failure path degrades to placeholder text, never an exception."""
    st.markdown(f"#### {title}")
    if img_rgb is None:
        st.info("Upload a target image to run inference.")
        return
    try:
        if key not in models:
            raise FileNotFoundError(errors.get(key, f"{MODEL_PATHS[key].name} not loaded"))
        df = predict_confidences(models[key], img_rgb)
        top = df["Confidence"].idxmax()
        st.metric("Predicted class", top, f"{df['Confidence'].max():.1%} confidence", delta_color="off")
        st.bar_chart(df, y="Confidence", color=BAR_COLORS[key], height=280)
    except Exception as exc:
        st.warning(f"Prediction unavailable - model weights not ready.\n\n`{exc}`")
        st.bar_chart(pd.DataFrame({"Confidence": [0.0] * len(CLASSES)},
                                  index=pd.Index(CLASSES, name="Class")), height=280)


# --------------------------------------------------------------------------- #
# Aggregate statistics block
# --------------------------------------------------------------------------- #
def measured_stats_markdown() -> str:
    """Per-domain mean accuracy from results/ood_summary.csv (written by ml/train_eval.py)."""
    try:
        s = pd.read_csv(ROOT / "results" / "ood_summary.csv")
        g = s.groupby("test_domain")[["baseline", "randomized"]].mean() * 100
        lines = ["| Test domain | Naive Baseline | Domain Randomized |", "|---|---|---|"]
        lines += [f"| {d} | {r.baseline:.1f}% | {r.randomized:.1f}% |" for d, r in g.iterrows()]
        return "\n".join(lines)
    except Exception:
        return "_Measured benchmark not found - run `python ml/train_eval.py` to generate it._"


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="OOD Domain Adaptation", page_icon=None, layout="wide")
    st.title("Out-of-Distribution Robustness: Naive vs Domain-Randomized SVM")
    st.caption("HOG + RBF-SVM classifiers evaluated zero-shot on a synthetically corrupted target-domain image.")

    models, errors = load_models()
    if len(models) < 2:
        st.warning("Model weights missing or unreadable - the layout is live but predictions are disabled. "
                   "Run `python ml/train_eval.py`, then press **Reload models**.")

    col_ctrl, col_live = st.columns([1, 2], gap="large")

    # ---------------- Column 1: control panel ----------------
    with col_ctrl:
        st.subheader("Control panel")
        uploaded = st.file_uploader("Target image", type=["png", "jpg", "jpeg", "bmp", "webp"],
                                    help="A single shape: Circle, Square, Triangle or Star.")
        corruption = st.selectbox("Target-domain corruption", CORRUPTIONS)
        severity = st.slider("Severity (0 = none, 5 = extreme)", 0.0, 5.0, 2.0, 0.5)
        if st.button("Reload models"):
            load_models.clear()
            st.rerun()

        st.markdown("### Aggregate accuracy")
        st.markdown(
            f"**Paper-reported (zero-shot, aggregate):**  \n"
            f"- Naive Baseline: **~{PAPER_STATS['Naive Baseline']}%**  \n"
            f"- Domain Randomized: **~{PAPER_STATS['Domain Randomized']}%**"
        )
        st.markdown("**Measured in this repository** (mean over severities 1-5):")
        st.markdown(measured_stats_markdown())

    # ---------------- Column 2: live evaluation ----------------
    with col_live:
        st.subheader("Live evaluation")
        original, corrupted = None, None
        if uploaded is not None:
            try:
                original = load_uploaded_image(uploaded)
                corrupted = apply_corruption(original, corruption, severity)
            except Exception as exc:
                st.error(f"Could not process this image: {exc}")

        if corrupted is not None:
            img_a, img_b = st.columns(2)
            show_image(img_a, original, "Original (resized to 96x96)")
            show_image(img_b, corrupted, f"{corruption} - severity {severity:g}")
        else:
            st.info("Upload a target image in the control panel to begin.")

        left, right = st.columns(2)
        with left:
            render_prediction_panel("Naive Baseline", "baseline", models, errors, corrupted)
        with right:
            render_prediction_panel("Domain Randomized", "randomized", models, errors, corrupted)


if __name__ == "__main__":
    main()
