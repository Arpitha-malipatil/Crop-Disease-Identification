import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image
import tensorflow as tf
import os

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Identifier",
    page_icon="🌱",
    layout="centered"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Jost:wght@300;400;500;600;700&display=swap');

:root {
    --bg:          #0e1117;
    --surface:     #161b24;
    --surface2:    #1e2636;
    --border:      #2a3548;
    --border2:     #344060;
    --text:        #e8edf5;
    --text-sub:    #8896b0;
    --text-dim:    #4d5e7a;
    --green:       #3ddc84;
    --green-dim:   #1a6640;
    --green-glow:  rgba(61,220,132,0.15);
    --red:         #ff6b6b;
    --red-dim:     #6b1a1a;
    --red-glow:    rgba(255,107,107,0.15);
    --amber:       #ffb347;
    --amber-dim:   rgba(255,179,71,0.12);
    --teal:        #38c9c9;
    --accent:      #3ddc84;
}

html, body, [class*="css"] {
    font-family: 'Jost', sans-serif;
    color: var(--text);
}

/* ── Page background ── */
.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(61,220,132,0.07) 0%, transparent 70%);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Global text overrides ── */
p, div, span, label { color: var(--text); }

/* ══════════════════ HERO ══════════════════ */
.hero {
    padding: 3rem 0 2rem;
    text-align: center;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--green-glow);
    border: 1px solid var(--green-dim);
    color: var(--green);
    font-family: 'Playfair Display', serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.3rem 0.85rem;
    border-radius: 99px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0 0 0.7rem;
}
.hero h1 span { color: var(--green); }
.hero p {
    color: var(--text-sub);
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
    line-height: 1.6;
}

/* ══════════════════ DIVIDER ══════════════════ */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.6rem 0;
}

/* ══════════════════ SECTION LABEL ══════════════════ */
.section-label {
    font-family: 'Playfair Display', serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-sub);
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ══════════════════ UPLOAD WIDGET ══════════════════ */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border2) !important;
    border-radius: 14px !important;
    background: var(--surface) !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--green) !important;
    background: rgba(61,220,132,0.03) !important;
}
[data-testid="stFileUploader"] * {
    color: var(--text-sub) !important;
}
[data-testid="stFileUploader"] svg {
    fill: var(--text-dim) !important;
}
/* Upload text */
[data-testid="stFileUploaderDropzoneInstructions"] div span {
    color: var(--text) !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div small {
    color: var(--text-sub) !important;
}

/* ══════════════════ BUTTON ══════════════════ */
.stButton > button {
    background: var(--green) !important;
    color: #0a1a0e !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-family: 'Jost', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(61,220,132,0.25) !important;
}
.stButton > button:hover {
    background: #5deea0 !important;
    box-shadow: 0 6px 28px rgba(61,220,132,0.4) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ══════════════════ IMAGE ══════════════════ */
div[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid var(--border);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

/* ══════════════════ RESULT CARD ══════════════════ */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin-top: 1.4rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 18px 18px 0 0;
}
.result-card.healthy-card  { border-color: var(--green-dim); }
.result-card.healthy-card::before { background: linear-gradient(90deg, var(--green), #1df0a0); }
.result-card.disease-card  { border-color: #6b2a2a; }
.result-card.disease-card::before { background: linear-gradient(90deg, var(--red), #ff9a6b); }

/* Status pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.85rem;
    border-radius: 99px;
    font-family: 'Playfair Display', serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.pill-healthy {
    background: var(--green-glow);
    color: var(--green);
    border: 1px solid var(--green-dim);
}
.pill-disease {
    background: var(--red-glow);
    color: var(--red);
    border: 1px solid var(--red-dim);
}

/* Plant / disease names */
.result-plant-label {
    font-family: 'Playfair Display', serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.15rem;
}
.result-plant-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-sub);
    margin-bottom: 1rem;
}
.result-disease-label {
    font-family: 'Playfair Display', serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.2rem;
}
.result-disease-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 1.6rem;
    color: var(--text);
}
.result-disease-name.is-healthy { color: var(--green); }
.result-disease-name.is-disease { color: #ff8a8a; }

/* Confidence block */
.conf-row {
    display: flex;
    align-items: flex-end;
    gap: 1.2rem;
    margin-bottom: 0.7rem;
}
.conf-block {}
.conf-block-label {
    font-family: 'Playfair Display', serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.15rem;
}
.conf-block-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    color: var(--text);
}
.conf-block-value.high   { color: var(--green); }
.conf-block-value.medium { color: var(--amber); }
.conf-block-value.low    { color: var(--red); }

.conf-bar-wrap { flex: 1; padding-bottom: 0.55rem; }
.conf-bar-bg {
    background: var(--surface2);
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}
.fill-high   { background: linear-gradient(90deg, #28b060, var(--green)); }
.fill-medium { background: linear-gradient(90deg, #c87a10, var(--amber)); }
.fill-low    { background: linear-gradient(90deg, #b03030, var(--red)); }

/* Meta chips */
.meta-chips {
    display: flex;
    gap: 0.55rem;
    flex-wrap: wrap;
    margin-top: 1.4rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
}
.meta-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.28rem 0.7rem;
    font-size: 0.72rem;
    color: var(--text-sub);
}
.meta-chip strong {
    color: var(--text);
    font-weight: 600;
}

/* ══════════════════ TOP-5 SECTION ══════════════════ */
.top5-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    margin-top: 1rem;
}
.top5-heading {
    font-family: 'Playfair Display', serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-sub);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.top5-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

.bar-row { margin-bottom: 1rem; }
.bar-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.3rem;
    gap: 0.5rem;
}
.bar-label {
    font-family: 'Jost', sans-serif;
    font-size: 0.84rem;
    color: var(--text-sub);
    font-weight: 400;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.bar-label.is-top { color: var(--text); font-weight: 600; }
.bar-pct {
    font-family: 'Jost', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text-sub);
    white-space: nowrap;
    flex-shrink: 0;
}
.bar-pct.is-top { color: var(--green); }
.bar-track {
    background: var(--surface2);
    border-radius: 99px;
    height: 9px;
    overflow: hidden;
    border: 1px solid var(--border);
    width: 100%;
}
.bar-fill-green { height: 100%; border-radius: 99px; background: linear-gradient(90deg,#1a8a50,var(--green)); }
.bar-fill-dim   { height: 100%; border-radius: 99px; background: linear-gradient(90deg,#2a3548,#3a4a5a); }

/* ══════════════════ LOW CONF ADVISORY ══════════════════ */
.advisory {
    background: var(--amber-dim);
    border: 1px solid rgba(255,179,71,0.25);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-top: 1rem;
    font-size: 0.84rem;
    color: #f0c070;
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
    line-height: 1.5;
}

/* ══════════════════ STREAMLIT OVERRIDES ══════════════════ */
/* Spinner */
.stSpinner > div { color: var(--green) !important; }
/* Alert / info boxes */
div[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 32
MODEL_PATH   = "plant_disease_cnn.h5"
ENCODER_PATH = "pca_encoder.pkl"

# Exact 38 classes in encoder order (confirmed from encoder.classes_)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

def format_label(raw: str):
    parts   = raw.split("___")
    plant   = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else ""
    return plant, disease

def conf_class(pct: float) -> str:
    if pct >= 70: return "high"
    if pct >= 45: return "medium"
    return "low"

# ─── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    errors, cnn_model, enc = [], None, None
    if not os.path.exists(MODEL_PATH):
        errors.append(f"CNN model not found: `{MODEL_PATH}`")
    else:
        try:
            cnn_model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            errors.append(f"Failed to load CNN: {e}")
    if not os.path.exists(ENCODER_PATH):
        errors.append(f"Encoder not found: `{ENCODER_PATH}`")
    else:
        try:
            with open(ENCODER_PATH, "rb") as f:
                enc = pickle.load(f)["encoder"]
        except Exception as e:
            errors.append(f"Failed to load encoder: {e}")
    return cnn_model, enc, errors

cnn_model, encoder, load_errors = load_models()

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🌱 AI Plant Diagnostics</div>
  <h1>Plant Disease <span>Identifier</span></h1>
  <p>Upload a leaf photo for CNN-powered disease detection<br>across 38 plant species &amp; conditions</p>
</div>
""", unsafe_allow_html=True)

# ─── Error display ────────────────────────────────────────────────────────────
if load_errors:
    for err in load_errors:
        st.error(f"⚠️ {err}")
    st.info("Place `plant_disease_cnn.h5` and `pca_encoder.pkl` in the same folder as `app.py`, then run `streamlit run app.py`.")
    st.stop()

# ─── Upload ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Upload Leaf Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    _, col_c, _ = st.columns([1, 5, 1])
    with col_c:
        st.image(image, use_column_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.button("🔬  Run Diagnosis"):
        with st.spinner("Analysing leaf..."):
            img = np.array(image)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            probs      = cnn_model.predict(img, verbose=0)[0]
            pred_idx   = int(np.argmax(probs))
            confidence = float(probs[pred_idx]) * 100

        if pred_idx >= len(CLASS_NAMES):
            st.error("Prediction index out of range.")
            st.stop()

        raw_label           = CLASS_NAMES[pred_idx]
        plant_name, disease = format_label(raw_label)
        is_healthy          = "healthy" in raw_label.lower()
        display_disease     = disease if disease else raw_label
        cc                  = conf_class(confidence)
        card_cls            = "healthy-card" if is_healthy else "disease-card"
        pill_cls            = "pill-healthy" if is_healthy else "pill-disease"
        pill_icon           = "✦" if is_healthy else "⚠"
        pill_txt            = f"{pill_icon} Healthy" if is_healthy else f"{pill_icon} Disease Detected"
        dis_cls             = "is-healthy" if is_healthy else "is-disease"

        # ── Result card ───────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-card {card_cls}">

          <span class="status-pill {pill_cls}">{pill_txt}</span>

          <div class="result-plant-label">Plant Species</div>
          <div class="result-plant-name">{plant_name}</div>

          <div class="result-disease-label">Diagnosis</div>
          <div class="result-disease-name {dis_cls}">{display_disease}</div>

          <div class="conf-row">
            <div class="conf-block">
              <div class="conf-block-label">Confidence</div>
              <div class="conf-block-value {cc}">{confidence:.1f}%</div>
            </div>
            <div class="conf-bar-wrap">
              <div class="conf-bar-bg">
                <div class="conf-bar-fill fill-{cc}" style="width:{min(confidence,100):.1f}%"></div>
              </div>
            </div>
          </div>

          <div class="meta-chips">
            <div class="meta-chip">Model <strong>CNN</strong></div>
            <div class="meta-chip">Input <strong>{IMAGE_SIZE}×{IMAGE_SIZE} px</strong></div>
            <div class="meta-chip">Classes <strong>{len(CLASS_NAMES)}</strong></div>
            <div class="meta-chip">Status <strong>{'Healthy ✓' if is_healthy else 'Diseased ✗'}</strong></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Low confidence advisory
        if confidence < 60:
            st.markdown("""
            <div class="advisory">
              <span>💡</span>
              <span><strong>Low confidence result.</strong> For better accuracy, use a clear well-lit photo
              of a single leaf placed against a plain background, with the disease symptoms clearly visible.</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Top 5 predictions ─────────────────────────────────────────────────
        top5_idx   = np.argsort(probs)[::-1][:5]
        top5_probs = probs[top5_idx]

        rows_html = ""
        for rank, (idx, prob) in enumerate(zip(top5_idx, top5_probs)):
            pct      = float(prob) * 100
            _, dis   = format_label(CLASS_NAMES[idx])
            label    = dis if dis else CLASS_NAMES[idx]
            is_top   = (idx == pred_idx)
            lbl_cls  = "is-top" if is_top else ""
            fill_cls = "bar-fill-green" if is_top else "bar-fill-dim"
            pct_cls  = "is-top" if is_top else ""
            rows_html += (
                f'<div class="bar-row">'
                f'<div class="bar-header">'
                f'<div class="bar-label {lbl_cls}">{label}</div>'
                f'<div class="bar-pct {pct_cls}">{pct:.1f}%</div>'
                f'</div>'
                f'<div class="bar-track">'
                f'<div class="{fill_cls}" style="width:{min(pct,100):.1f}%"></div>'
                f'</div>'
                f'</div>'
            )

        st.markdown(
            '<div class="top5-card">'
            '<div class="top5-heading">Top 5 Predictions</div>'
            + rows_html +
            '</div>',
            unsafe_allow_html=True
        )
