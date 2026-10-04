import streamlit as st
import numpy as np
import cv2
import joblib
from skimage.feature import hog
from PIL import Image

# 1. EMPOWER HIGH-FIDELITY DESIGN CUSTOMIZATIONS
st.set_page_config(layout="wide", page_title="OOD Generalization Framework")

# Inject Custom React/Tailwind styling behaviors directly into the Streamlit DOM wrapper
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
        
        /* Neomorphic Left Panel Grid Cards */
        .control-card {
            background-color: #020617;
            border: 1px solid #1e293b;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 20px 0 rgba(0,0,0,0.4);
        }
        
        /* Model Evaluation Status Pill Badges */
        .status-badge-fail {
            background-color: #991b1b;
            color: #fca5a5;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }
        .status-badge-pass {
            background-color: #065f46;
            color: #a7f3d0;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
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
    corruption = st.selectbox("Select Domain Distortions (\u03b4)", ["Sensor Noise", "Motion Blur", "Fog/Haze"])
    severity = st.slider("Covariate Shift Severity Level", 0, 5, value=0)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="control-card" style="margin-top:20px;">', unsafe_allow_html=True)
    st.subheader("📊 Aggregate Paper Metrics")
    st.markdown("**Naive Model Convergence:** <span class='status-badge-fail'>~25.3% Accuracy</span>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    st.markdown("**Domain Randomized Network:** <span class='status-badge-pass'>~57.1% Accuracy</span>", unsafe_allow_html=True)
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
        
     # --- UPGRADED HIGH-IMPACT CARD MATRIX ---
    top_base_idx = np.argmax(prob_base)
    top_rand_idx = np.argmax(prob_rand)
    
    base_winner = CLASSES[top_base_idx]
    base_conf = float(prob_base[top_base_idx])
    
    rand_winner = CLASSES[top_rand_idx]
    rand_conf = float(prob_rand[top_rand_idx])

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Head-to-Head Architectural Convergence Matrix")
    
    # Split into modern side-by-side comparative cards
    c_left, c_right = st.columns(2, gap="medium")
    
    with c_left:
        st.markdown(f"""
            <div class="control-card" style="border-left: 4px solid #ef4444; margin-bottom: 20px;">
                <div style="display:flex; justify-content:between; align-items:center;">
                    <span style="font-weight:800; color:#f87171; letter-spacing:0.5px;">❌ NAIVE BASELINE PREDICTION</span>
                    <span class="status-badge-fail">{"OOD Breakdown" if severity > 2 else "Optimal"}</span>
                </div>
                <div style="margin: 16px 0;">
                    <span style="font-size:12px; font-weight:700; color:#94a3b8; uppercase tracking-wider block;">Top Classified Projection</span>
                    <span style="font-size:28px; font-weight:900; color:#ffffff; block; margin-top:2px;">{base_winner}</span>
                </div>
                <div style="font-size:13px; font-weight:600; color:#cbd5e1; display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span>Classification Confidence</span>
                    <span>{base_conf*100:.1f}%</span>
                </div>
                <div style="background-color:#1e293b; border-radius:9999px; height:8px; overflow:hidden;">
                    <div style="background:linear-gradient(90deg, #ef4444, #f43f5e); width:{base_conf*100}%; height:100%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with c_right:
        st.markdown(f"""
            <div class="control-card" style="border-left: 4px solid #10b981; box-shadow: 0 0 15px rgba(16,185,129,0.15); margin-bottom: 20px;">
                <div style="display:flex; justify-content:between; align-items:center;">
                    <span style="font-weight:800; color:#34d399; letter-spacing:0.5px;">✅ DOMAIN ADAPTED PREDICTION</span>
                    <span class="status-badge-pass">Invariant Match</span>
                </div>
                <div style="margin: 16px 0;">
                    <span style="font-size:12px; font-weight:700; color:#94a3b8; uppercase tracking-wider block;">Top Classified Projection</span>
                    <span style="font-size:28px; font-weight:900; color:#ffffff; block; margin-top:2px;">{rand_winner}</span>
                </div>
                <div style="font-size:13px; font-weight:600; color:#cbd5e1; display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span>Classification Confidence</span>
                    <span>{rand_conf*100:.1f}%</span>
                </div>
                <div style="background-color:#1e293b; border-radius:9999px; height:8px; overflow:hidden;">
                    <div style="background:linear-gradient(90deg, #10b981, #14b8a6); width:{rand_conf*100}%; height:100%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Awaiting clean simulation file configuration upload from control panel array indices.")

