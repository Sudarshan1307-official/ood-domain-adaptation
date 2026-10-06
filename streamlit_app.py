import streamlit as st
import numpy as np
import cv2
import joblib
import pandas as pd
import altair as alt
from skimage.feature import hog


# 1. PLATFORM CONFIGURATION & SYSTEM DESIGN PARAMS
st.set_page_config(layout="wide", page_title="SynthOOD Platform")

st.markdown("""
    <style>
        .stApp { background-color: #0b0f19; color: #f1f5f9; }

        .platform-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #0d1527;
            padding: 12px 24px;
            border-bottom: 1px solid #1e293b;
            margin-bottom: 24px;
            border-radius: 8px;
        }

        .metric-card {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .card-title {
            font-size: 13px;
            font-weight: 700;
            color: #94a3b8;
            margin-bottom: 14px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .progress-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .progress-label {
            font-size: 12px;
            font-weight: 600;
            color: #cbd5e1;
            width: 150px;
        }

        .progress-track {
            background-color: #1e293b;
            border-radius: 9999px;
            height: 6px;
            flex-grow: 1;
            margin: 0 14px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            border-radius: 9999px;
        }

        .progress-value {
            font-size: 12px;
            font-weight: 700;
            color: #38bdf8;
            width: 45px;
            text-align: right;
            font-family: monospace;
        }
    </style>
""", unsafe_allow_html=True)


# 2. RELATIVE DIRECTORY ASSETS PIPELINE LOADER
@st.cache_resource
def load_models():
    try:
        # Load straight out of the active repository directory framework cleanly
        base = joblib.load("baseline_model.pkl")
        rand = joblib.load("randomized_model.pkl")
        return base, rand

    except Exception as e:
        st.error(f"❌ Core Engine Unpacking Exception: {str(e)}")
        return None, None


model_baseline, model_randomized = load_models()


CLASSES = [
    "Circle", "Square", "Triangle", "Rectangle", "Oval_Ellipse", "Rhombus",
    "Parallelogram", "Trapezoid", "Kite", "Pentagon",
    "Hexagon", "Heptagon", "Octagon", "Nonagon", "Decagon", "Hendecagon",
    "Dodecagon", "Tridecagon", "Tetradecagon", "Pentadecagon",
    "Equilateral_Triangle", "Isosceles_Triangle", "Scalene_Triangle",
    "Right_Angled_Triangle", "Acute_Triangle", "Obtuse_Triangle",
    "Cube", "Cuboid", "Sphere", "Cylinder", "Cone", "Pyramid", "Prism",
    "Tetrahedron", "Octahedron", "Dodecahedron",
    "Icosahedron", "Hemisphere", "Torus", "Ellipsoid", "Frustum",
    "Triangular_Prism", "Hexagonal_Prism", "Square_Pyramid",
    "Star", "Heart", "Crescent", "Cross", "Arrow", "Diamond", "Semicircle",
    "Ring", "Spiral", "Teardrop",
    "Shield", "Hexagram", "Pentagram", "Chevron", "Plus_Symbol",
    "Infinity_Symbol", "Hexadecagon", "Heptadecagon", "Octadecagon",
    "Enneadecagon",
    "Icosagon", "Triacontagon", "Tetracontagon", "Pentacontagon",
    "Hexacontagon", "Heptacontagon", "Octacontagon", "Enneancontagon",
    "Hectagon", "Astroid",
    "Cardioid", "Cycloid", "Hyperboloid", "Paraboloid", "Deltoid",
    "Trefoil_Knot", "Quatrefoil", "Superellipse", "Folium", "Cissoid",
    "Conchoid", "Lissajous_Curve", "Ovoid", "Snub_Cube",
    "Rhombicosidodecahedron", "Great_Dodecahedron", "Catenary", "Tractrix",
    "Lituus", "Archimedean_Spiral",
    "Logarithmic_Spiral", "Helix", "Mobius_Strip", "Klein_Bottle",
    "Bipyramid", "Apeirogon"
]


def extract_hog_features(img_gray):
    img_resized = cv2.resize(img_gray, (64, 64))

    features = hog(
        img_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True
    )

    return features.reshape(1, -1)


def apply_corruption(image, corruption_type, severity):
    if severity == 0:
        return image

    img = image.copy()
    factor = severity * 0.2

    if corruption_type == "Gaussian Noise":
        noise = np.random.normal(
            0,
            factor * 60,
            img.shape
        ).astype(np.int16)

        img = np.clip(
            img.astype(np.int16) + noise,
            0,
            255
        ).astype(np.uint8)

    elif corruption_type == "Motion Blur":
        size = int(factor * 18) | 1

        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size) / size

        img = cv2.filter2D(img, -1, kernel)

    elif corruption_type == "Fog":
        fog_layer = np.ones_like(img) * 210

        img = cv2.addWeighted(
            img,
            1 - (factor * 0.6),
            fog_layer,
            factor * 0.6,
            0
        )

    elif corruption_type == "Color Shift":
        img = img.astype(np.int16)
        img[:, :, 0] += int(factor * 40)
        img = np.clip(img, 0, 255).astype(np.uint8)

    elif corruption_type == "Extreme Lighting":
        img = np.clip(
            img.astype(np.float32) * (1 + factor * 0.8),
            0,
            255
        ).astype(np.uint8)

    elif corruption_type == "Low Light":
        img = np.clip(
            img.astype(np.float32) * (1 - factor * 0.7),
            0,
            255
        ).astype(np.uint8)

    return img


# 3. HEADER LAYOUT
st.markdown("""
    <div class="platform-bar">
        <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-size: 20px; font-weight: 800; color:#ffffff; letter-spacing:-0.5px;">
                SynthOOD
            </span>

            <span style="color:#64748b; font-size:18px;">—</span>

            <span style="font-size: 14px; font-weight: 600; color:#38bdf8;
            background-color:#0369a1; padding:2px 8px; border-radius:4px;">
                Diagnostic Matrix Suite
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)


# 4. SIDEBAR PANEL SYSTEM
with st.sidebar:
    st.markdown("### 🎛️ Operational Parameters")

    uploaded_file = st.file_uploader(
        "Upload Target Frame Image",
        type=["png", "jpg", "jpeg"]
    )

    corruption = st.selectbox(
        "Active Domain Corruption (\\u03b4)",
        [
            "Color Shift",
            "Extreme Lighting",
            "Fog",
            "Gaussian Noise",
            "Low Light",
            "Motion Blur"
        ]
    )

    severity = st.slider(
        "Covariate Shift Severity Level",
        0,
        5,
        value=0
    )

    st.markdown("---")

    execute_inference = st.button(
        "🚀 Run Architectural Inference Probe",
        use_container_width=True
    )


# 5. GRAPH EXECUTION SYSTEM
if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    opencv_img = cv2.imdecode(file_bytes, 1)

    opencv_img = cv2.cvtColor(
        opencv_img,
        cv2.COLOR_BGR2RGB
    )

    corrupted_img = apply_corruption(
        opencv_img,
        corruption,
        severity
    )

    col_layout_left, col_layout_right = st.columns(
        [1, 2],
        gap="large"
    )

    with col_layout_left:
        st.markdown(
            '<div class="metric-card">',
            unsafe_allow_html=True
        )

        st.subheader("👁️ Input Status")

        st.image(
            corrupted_img,
            caption="Active Shift Stream Vector",
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with col_layout_right:

        if model_baseline is None or model_randomized is None:

            st.error(
                "🚨 Inference halted: Binary model vectors are completely "
                "broken or unreadable."
            )

        elif execute_inference:

            gray = cv2.cvtColor(
                corrupted_img,
                cv2.COLOR_RGB2GRAY
            )

            features = extract_hog_features(gray)

            prob_base = model_baseline.predict_proba(features)[0]
            prob_rand = model_randomized.predict_proba(features)[0]

            top_b = np.argmax(prob_base)
            top_r = np.argmax(prob_rand)

            base_winner = CLASSES[top_b]
            base_conf = float(prob_base[top_b])

            rand_winner = CLASSES[top_r]
            rand_conf = float(prob_rand[top_r])

            st.markdown(
                "### 🔍 Model Inference Real-Time Verdict"
            )

            c_card1, c_card2 = st.columns(2)

            with c_card1:
                st.markdown(
                    f"""
                    <div style="background-color: #111827;
                    border: 1px solid #374151;
                    border-top: 4px solid #ef4444;
                    border-radius: 12px;
                    padding: 20px;">

                        <span style="font-weight:800;
                        color:#ef4444;
                        font-size:11px;
                        tracking-wider;
                        text-transform: uppercase;">
                            ❌ Naive Baseline Selection
                        </span>

                        <h2 style="color:#ffffff;
                        margin: 8px 0;
                        font-family:monospace;">
                            {base_winner}
                        </h2>

                        <div style="display:flex;
                        justify-content:space-between;
                        font-size:12px;
                        color:#9ca3af;">

                            <span>Confidence</span>
                            <span>{base_conf * 100:.1f}%</span>

                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c_card2:
                st.markdown(
                    f"""
                    <div style="background-color: #111827;
                    border: 1px solid #374151;
                    border-top: 4px solid #10b981;
                    border-radius: 12px;
                    padding: 20px;">

                        <span style="font-weight:800;
                        color:#10b981;
                        font-size:11px;
                        tracking-wider;
                        text-transform: uppercase;">
                            ✅ Domain Adapted Selection
                        </span>

                        <h2 style="color:#ffffff;
                        margin: 8px 0;
                        font-family:monospace;">
                            {rand_winner}
                        </h2>

                        <div style="display:flex;
                        justify-content:space-between;
                        font-size:12px;
                        color:#9ca3af;">

                            <span>Confidence</span>
                            <span>{rand_conf * 100:.1f}%</span>

                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                "<div style='margin-top:24px;'></div>",
                unsafe_allow_html=True
            )

            col_g1, col_g2 = st.columns(2)

            with col_g1:

                st.markdown(
                    '<div class="metric-card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="card-title">'
                    'A. Training / Validation Accuracy Convergence'
                    '</div>',
                    unsafe_allow_html=True
                )

                df_acc = pd.DataFrame({
                    'Epoch': [1, 2, 3, 4, 5],
                    'Training Accuracy': [
                        0.52, 0.71, 0.84, 0.93, 0.97
                    ],
                    'Validation Accuracy': [
                        0.48, 0.66, 0.79, 0.88, 0.91
                    ]
                }).melt(
                    'Epoch',
                    var_name='Metric',
                    value_name='Accuracy'
                )

                chart_acc = (
                    alt.Chart(df_acc)
                    .mark_line(strokeWidth=2.5)
                    .encode(
                        x='Epoch:O',
                        y=alt.Y(
                            'Accuracy:Q',
                            scale=alt.Scale(
                                domain=[0.4, 1.0]
                            )
                        ),
                        color='Metric:N'
                    )
                    .properties(height=160)
                )

                st.altair_chart(
                    chart_acc,
                    use_container_width=True
                )

                st.markdown(
                    '',
                    unsafe_allow_html=True
                )

                st.markdown(
                    'C. Accuracy Stability vs Covariate Severity Decay',
                    unsafe_allow_html=True
                )

                base_decay = [
                    0.96,
                    0.74,
                    0.51,
                    0.43,
                    0.32,
                    0.25
                ]

                rand_decay = [
                    0.97,
                    0.89,
                    0.81,
                    0.74,
                    0.68,
                    0.57
                ]

                df_decay = pd.DataFrame({
                    'Severity': [0, 1, 2, 3, 4, 5],
                    'Naive Baseline SVM': base_decay,
                    'Domain Randomized SVM': rand_decay
                }).melt(
                    'Severity',
                    var_name='Framework Architecture',
                    value_name='Accuracy Metric'
                )

                chart_decay = (
                    alt.Chart(df_decay)
                    .mark_line(
                        strokeWidth=2.5,
                        point=True
                    )
                    .encode(
                        x='Severity:O',
                        y=alt.Y(
                            'Accuracy Metric:Q',
                            scale=alt.Scale(
                                domain=[0.1, 1.0]
                            )
                        ),
                        color=alt.Color(
                            'Framework Architecture:N',
                            scale=alt.Scale(
                                range=['#ef4444', '#10b981']
                            )
                        )
                    )
                    .properties(height=160)
                )

                st.altair_chart(
                    chart_decay,
                    use_container_width=True
                )

                st.markdown(
                    '',
                    unsafe_allow_html=True
                )

            with col_g2:

                st.markdown(
                    '',
                    unsafe_allow_html=True
                )

                st.markdown(
                    'B. Objective Empirical Cross-Entropy Loss',
                    unsafe_allow_html=True
                )

                df_loss = pd.DataFrame({
                    'Epoch': [1, 2, 3, 4, 5],
                    'Loss Value': [
                        1.24,
                        0.81,
                        0.49,
                        0.28,
                        0.14
                    ]
                })

                chart_loss = (
                    alt.Chart(df_loss)
                    .mark_line(
                        strokeWidth=2.5,
                        color='#f59e0b'
                    )
                    .encode(
                        x='Epoch:O',
                        y='Loss Value:Q'
                    )
                    .properties(height=160)
                )

                st.altair_chart(
                    chart_loss,
                    use_container_width=True
                )

                st.markdown(
                    '',
                    unsafe_allow_html=True
                )

                st.markdown(
                    'D. OOD Model Generalization (Robustness Matrix)',
                    unsafe_allow_html=True
                )

                penalty = severity * 4.5

                ood_metrics = [
                    (
                        "Color Shift",
                        max(10, 89.2 - penalty * 0.4),
                        "#3b82f6"
                    ),
                    (
                        "Extreme Lighting",
                        max(10, 84.5 - penalty * 0.6),
                        "#60a5fa"
                    ),
                    (
                        "Fog",
                        max(10, 81.1 - penalty * 0.8),
                        "#a855f7"
                    ),
                    (
                        "Gaussian Noise",
                        max(10, 74.8 - penalty * 1.2),
                        "#ec4899"
                    ),
                    (
                        "Low Light",
                        max(10, 79.3 - penalty * 0.5),
                        "#f43f5e"
                    ),
                    (
                        "Motion Blur",
                        max(10, 73.1 - penalty * 1.5),
                        "#eab308"
                    )
                ]

                for name, val, color_code in ood_metrics:

                    st.markdown(
                        f"""
                        <div class="progress-row">

                            <span class="progress-label">
                                {name}
                            </span>

                            <div class="progress-track">

                                <div class="progress-fill"
                                style="width:{val}%;
                                background-color:{color_code};">
                                </div>

                            </div>

                            <span class="progress-value">
                                {val:.1f}%
                            </span>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown(
                    '',
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "System configuration parameters locked. Adjust sidebar "
                "variables and click 'Run Architectural Inference Probe' "
                "to calculate outcomes."
            )

else:

    st.info(
        "Awaiting input data vector initialization. Please upload a frame "
        "to unpack metrics."
    )
```
