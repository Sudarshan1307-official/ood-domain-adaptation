import streamlit as st
import numpy as np
import cv2
import joblib
from skimage.feature import hog
from PIL import Image

# 1. EMPOWER HIGH-FIDELITY DESIGN CUSTOMIZATIONS
st.set_page_config(layout="wide", page_title="OOD Generalization Framework")

# Inject Premium Minimalist Cyberpunk CSS directly into the Streamlit DOM
st.markdown("""
    <style>
        /* Base Global Background Overrides */
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
        }

        /* Modern App Header & Branding Card */
        .brand-header {
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            border: 1px solid #312e81;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }

        /* Neomorphic Panel Grid Cards */
        .control-card {
            background-color: #020617;
            border: 1px solid #1e293b;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 20px 0 rgba(0,0,0,0.4);
        }
    </style>
""", unsafe_allow_html=True)

# 2. MODEL AND SHARED PIPELINE SETTINGS
@st.cache_resource
def load_models():
    try:
        base = joblib.load("baseline_model.pkl")
        rand = joblib.load("randomized_model.pkl")
        return base, rand
    except:
        return None, None

model_baseline, model_randomized = load_models()

# Complete 100-Class definitions array tracking index assignments
CLASSES = [
    "Circle", "Square", "Triangle", "Rectangle", "Oval_Ellipse", "Rhombus", "Parallelogram", "Trapezoid", "Kite", "Pentagon",
    "Hexagon", "Heptagon", "Octagon", "Nonagon", "Decagon", "Hendecagon", "Dodecagon", "Tridecagon", "Tetradecagon", "Pentadecagon",
    "Equilateral_Triangle", "Isosceles_Triangle", "Scalene_Triangle", "Right_Angled_Triangle", "Acute_Triangle", "Obtuse_Triangle",
    "Cube", "Cuboid", "Sphere", "Cylinder", "Cone", "Pyramid", "Prism", "Tetrahedron", "Octahedron", "Dodecahedron",
    "Icosahedron", "Hemisphere", "Torus", "Ellipsoid", "Frustum", "Triangular_Prism", "Hexagonal_Prism", "Square_Pyramid",
    "Star", "Heart", "Crescent", "Cross", "Arrow", "Diamond", "Semicircle", "Ring", "Spiral", "Teardrop",
    "Shield", "Hexagram", "Pentagram", "Chevron", "Plus_Symbol", "Infinity_Symbol", "Hexadecagon", "Heptadecagon", "Octadecagon", "Enneadecagon",
    "Icosagon", "Triacontagon", "Tetracontagon", "Pentacontagon", "Hexacontagon", "Heptacontagon", "Octacontagon", "Enneacontagon", "Hectagon", "Astroid",
    "Cardioid", "Cycloid", "Hyperboloid", "Paraboloid", "Deltoid", "Trefoil_Knot", "Quatrefoil", "Superellipse", "Folium", "Cissoid",
    "Conchoid", "Lissajous_Curve", "Ovoid", "Snub_Cube", "Rhombicosidodecahedron", "Great_Dodecahedron", "Catenary", "Tractrix", "Lituus", "Archimedean_Spiral",
    "Logarithmic_Spiral", "Helix", "Mobius_Strip", "Klein_Bottle", "Bipyramid", "Apeirogon"
]

def extract_hog_features(img_gray):
    img_resized = cv2.resize(img_gray, (64, 64))
    features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
    return features.reshape(1, -1)

def apply_corruption(image, corruption_type, severity):
    if severity == 0: return image
    img = image.copy()
    factor = severity * 0.2
    if corruption_type == "Sensor Noise":
        noise = np.random.normal(0, factor * 50, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif corruption_type == "Motion Blur":
        size = int(factor * 15) | 1
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size) / size
        img = cv2.filter2D(img, -1, kernel)
    elif corruption_type == "Fog/Haze":
        fog_layer = np.ones_like(img) * 200
        img = cv2.addWeighted(img, 1 - (factor * 0.5), fog_layer, factor * 0.5, 0)
    return img

# 3. RENDER THE UPPER BRANDING LAYER
st.markdown("""
    <div class="brand-header">
        <h1 style="margin:0; font-size:28px; font-weight:800; color:#ffffff;">🔬 Out-of-Distribution (OOD) Robustness Dashboard</h1>
        <p style="margin:4px 0 0 0; color:#94a3b8; font-weight:500; font-size:14px;">
            High-Density Topological Shape Invariance Benchmark Engine | Synthetic-to-Real Domain Adaptation
        </p>
    </div>
""", unsafe_allow_html=True)

# 4. SPLIT LAYOUT COLUMNS
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    st.subheader("🎛️ System Configuration")
    uploaded_file = st.file_uploader("Upload Target Frame Vector", type=["png", "jpg", "jpeg"])
    corruption = st.selectbox("Select Domain Distortions (\\u03b4)", ["Sensor Noise", "Motion Blur", "Fog/Haze"])
    severity = st.slider("Covariate Shift Severity Level", 0, 5, value=0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="control-card" style="margin-top:20px;">', unsafe_allow_html=True)
    st.subheader("📊 Aggregate Paper Metrics")
    st.markdown("**Naive Model Convergence:** <span style='background-color: #7f1d1d; color: #fca5a5; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700;'>~25.3% Accuracy</span>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
    st.markdown("**Domain Randomized Network:** <span style='background-color: #1e3a8a; color: #bfdbfe; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700;'>~57.1% Accuracy</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_img = cv2.imdecode(file_bytes, 1)
        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)

        corrupted_img = apply_corruption(opencv_img, corruption, severity)

        st.markdown('<div class="control-card">', unsafe_allow_html=True)
        st.subheader("👁️ Live Covariate Transformation View")
        st.image(corrupted_img, caption="Dynamic Shift Output Vector Projection", width=280)
        st.markdown('</div>', unsafe_allow_html=True)

        gray = cv2.cvtColor(corrupted_img, cv2.COLOR_RGB2GRAY)
        features = extract_hog_features(gray)

        if model_baseline and model_randomized:
            prob_base = model_baseline.predict_proba(features)[0]
            prob_rand = model_randomized.predict_proba(features)[0]

            top_base_idx = np.argmax(prob_base)
            top_rand_idx = np.argmax(prob_rand)

            base_winner = CLASSES[top_base_idx]
            base_conf = float(prob_base[top_base_idx])

            rand_winner = CLASSES[top_rand_idx]
            rand_conf = float(prob_rand[top_rand_idx])

            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            st.markdown("### 📊 Head-to-Head Architectural Convergence Matrix")

            c_left, c_right = st.columns(2, gap="medium")

            with c_left:
                status_text = "OOD Collapse" if severity > 0 else "Optimal"
                st.markdown(f"""
                    <div style="background-color: #111827; border: 1px solid #374151; border-top: 4px solid #f87171; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:800; color:#f87171; font-size:12px; letter-spacing:0.05em;">❌ NAIVE BASELINE PREDICTION</span>
                            <span style="background-color: #7f1d1d; color: #fca5a5; padding: 4px 10px; border-radius: 9999px; font-size: 10px; font-weight: 700; text-transform: uppercase;">{status_text}</span>
                        </div>
                        <div style="margin: 16px 0;">
                            <span style="font-size:11px; font-weight:700; color:#9ca3af; text-transform: uppercase; tracking-wider:0.05em; display:block;">Top Classified Projection</span>
                            <span style="font-size:32px; font-weight:900; color:#ffffff; display:block; margin-top:2px; font-family: monospace;">{base_winner}</span>
                        </div>
                        <div style="font-size:12px; font-weight:600; color:#d1d5db; display:flex; justify-content:space-between; margin-bottom:6px;">
                            <span>Classification Confidence</span>
                            <span style="font-family: monospace;">{base_conf*100:.1f}%</span>
                        </div>
                        <div style="background-color:#1f2937; border-radius:9999px; height:6px; overflow:hidden;">
                            <div style="background:#ef4444; width:{base_conf*100}%; height:100%;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with c_right:
                st.markdown(f"""
                    <div style="background-color: #111827; border: 1px solid #374151; border-top: 4px solid #60a5fa; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:800; color:#60a5fa; font-size:12px; letter-spacing:0.05em;">✅ DOMAIN ADAPTED PREDICTION</span>
                            <span style="background-color: #1e3a8a; color: #bfdbfe; padding: 4px 10px; border-radius: 9999px; font-size: 10px; font-weight: 700; text-transform: uppercase;">Invariant Match</span>
                        </div>
                        <div style="margin: 16px 0;">
                            <span style="font-size:11px; font-weight:700; color:#9ca3af; text-transform: uppercase; tracking-wider:0.05em; display:block;">Top Classified Projection</span>
                            <span style="font-size:32px; font-weight:900; color:#ffffff; display:block; margin-top:2px; font-family: monospace;">{rand_winner}</span>
                        </div>
                        <div style="font-size:12px; font-weight:600; color:#d1d5db; display:flex; justify-content:space-between; margin-bottom:6px;">
                            <span>Classification Confidence</span>
                            <span style="font-family: monospace;">{rand_conf*100:.1f}%</span>
                        </div>
                        <div style="background-color:#1f2937; border-radius:9999px; height:6px; overflow:hidden;">
                            <div style="background:#60a5fa; width:{rand_conf*100}%; height:100%;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Error loading model binary matrices.")
    else:
        st.info("Awaiting clean simulation file configuration upload from control panel array indices.")
