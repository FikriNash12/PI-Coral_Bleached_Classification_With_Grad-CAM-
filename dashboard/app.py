import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="CoralSense — Deteksi Pemutihan Karang",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0f0d; color: #e8ede9; }
.hero-wrapper {
    background: linear-gradient(135deg, #0d2018 0%, #0a1a12 40%, #071510 100%);
    border: 1px solid #1a3d28; border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem; margin-bottom: 2.5rem;
    position: relative; overflow: hidden;
}
.hero-wrapper::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(32,180,100,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800; color: #e8ede9; line-height: 1.1; margin: 0 0 0.5rem 0; letter-spacing: -1px; }
.hero-title span { color: #20b464; }
.hero-sub { font-size: 1.05rem; color: #8aab92; font-weight: 300; margin: 0 0 1.5rem 0; max-width: 560px; line-height: 1.6; }
.badge-row { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.badge { background: rgba(32,180,100,0.12); border: 1px solid rgba(32,180,100,0.3); color: #20b464; padding: 0.3rem 0.85rem; border-radius: 999px; font-size: 0.78rem; font-weight: 500; }
.section-label { font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; color: #20b464; margin-bottom: 0.4rem; }
.section-title { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; color: #e8ede9; margin: 0 0 1.5rem 0; letter-spacing: -0.3px; }
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2.5rem; }
.metric-card { background: #0d1a12; border: 1px solid #1a3d28; border-radius: 14px; padding: 1.2rem 1.4rem; text-align: center; }
.metric-val { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #20b464; line-height: 1; margin-bottom: 0.3rem; }
.metric-lbl { font-size: 0.78rem; color: #6b8c74; font-weight: 400; }
.pred-bleached { background: linear-gradient(135deg, #2d0f0f, #1a0a0a); border: 1px solid #7f1d1d; border-radius: 12px; padding: 1rem 1.4rem; text-align: center; }
.pred-healthy { background: linear-gradient(135deg, #0d2018, #071510); border: 1px solid #1a3d28; border-radius: 12px; padding: 1rem 1.4rem; text-align: center; }
.pred-label { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 800; margin-bottom: 0.2rem; }
.pred-prob { font-size: 0.85rem; color: #8aab92; font-weight: 300; }
.info-box { background: #0d1a12; border-left: 3px solid #20b464; border-radius: 0 12px 12px 0; padding: 1.2rem 1.4rem; margin-top: 1rem; font-size: 0.88rem; color: #8aab92; line-height: 1.7; }
.divider { border: none; border-top: 1px solid #1a3d28; margin: 2.5rem 0; }
.upload-hint { font-size: 0.82rem; color: #4d6e56; margin-top: 0.5rem; font-style: italic; }
[data-testid="stFileUploader"] { background: #0d1a12; border: 2px dashed #1a3d28; border-radius: 14px; padding: 0.5rem; }
[data-testid="stFileUploader"]:hover { border-color: #20b464; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
IMG_SIZE = (128, 128)
MODEL_PATH = "dashboard/cnn_coral_bleaching.keras"

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def preprocess_image(img_array):
    img_pil = Image.fromarray(img_array)
    img_pil = img_pil.resize(IMG_SIZE, Image.BILINEAR)
    return np.array(img_pil)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='last_conv_layer'):
    img_tensor = tf.cast(img_array, tf.float32)
    layer_names = [l.name for l in model.layers]
    conv_idx = layer_names.index(last_conv_layer_name)
    pre_conv_layers = model.layers[:conv_idx + 1]
    post_conv_layers = model.layers[conv_idx + 1:]

    with tf.GradientTape() as tape:
        x = img_tensor
        for layer in pre_conv_layers:
            x = layer(x)
        conv_output = x
        tape.watch(conv_output)
        for layer in post_conv_layers:
            x = layer(x)
        preds = x
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def estimate_bleaching_percentage(img):
    r, g, b = img[:,:,0]/255.0, img[:,:,1]/255.0, img[:,:,2]/255.0
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    s = np.where(maxc != 0, (maxc - minc) / maxc, 0)
    v = maxc
    mask = (s < (60/255.0)) & (v > (170/255.0))
    return (np.sum(mask) / mask.size) * 100

def predict_and_visualize(img_rgb, model):
    img = preprocess_image(img_rgb)
    img_array = np.expand_dims(img, axis=0)
    img_tensor = tf.cast(img_array, tf.float32)

    pred_prob = float(model.predict(img_tensor, verbose=0)[0][0])
    pred_label = "Bleached" if pred_prob > 0.5 else "Healthy"
    bleach_pct = estimate_bleaching_percentage(img)

    heatmap = make_gradcam_heatmap(img_tensor, model)
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE, Image.BILINEAR)
    heatmap_resized = np.array(heatmap_pil) / 255.0
    heatmap_colored = np.uint8(cm.jet(heatmap_resized)[:, :, :3] * 255)
    superimposed = np.uint8(img * 0.6 + heatmap_colored * 0.4)

    return pred_label, pred_prob, bleach_pct, img, heatmap_resized, superimposed

def show_result(pred_label, pred_prob, bleach_pct, img, heatmap, superimposed):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img, caption="Gambar Asli", use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor('#0a0f0d')
        ax.imshow(heatmap, cmap='jet')
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.caption("Grad-CAM Heatmap")
    with col3:
        st.image(superimposed, caption="Superimposed", use_container_width=True)

    if pred_label == "Bleached":
        st.markdown(f"""
        <div class="pred-bleached">
            <div class="pred-label" style="color:#f87171;">🔴 {pred_label}</div>
            <div class="pred-prob">Probabilitas Bleaching: <b>{pred_prob:.4f}</b> &nbsp;|&nbsp; Estimasi Area Putih: <b>{bleach_pct:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        ⚠️ <b>Karang terdeteksi mengalami pemutihan (bleaching).</b> Pemutihan karang terjadi ketika karang melepaskan alga simbiotik (zooxanthellae) akibat tekanan suhu air laut yang meningkat. Area yang ditandai heatmap merah merupakan region yang menjadi fokus model dalam mengambil keputusan.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="pred-healthy">
            <div class="pred-label" style="color:#4ade80;">🟢 {pred_label}</div>
            <div class="pred-prob">Probabilitas Bleaching: <b>{pred_prob:.4f}</b> &nbsp;|&nbsp; Estimasi Area Putih: <b>{bleach_pct:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        ✅ <b>Karang terdeteksi dalam kondisi sehat.</b> Karang sehat memiliki warna yang beragam karena keberadaan zooxanthellae yang menghasilkan pigmen. Heatmap menunjukkan area tekstur dan warna yang menjadi indikator kesehatan karang bagi model.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# SAMPLE IMAGES
# ─────────────────────────────────────────
SAMPLE_IMAGES = [
    {"path": "dashboard/samples/noaa_healthy.jpg",  "label": "Healthy",  "source": "NOAA-PIFSC"},
    {"path": "dashboard/samples/cs_healthy.jpg",    "label": "Healthy",  "source": "Coralscapes"},
    {"path": "dashboard/samples/noaa_bleached.jpg", "label": "Bleached", "source": "NOAA-PIFSC"},
    {"path": "dashboard/samples/cs_bleached.jpg",   "label": "Bleached", "source": "Coralscapes"},
]

# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────

# HERO
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-title">Coral<span>Sense</span></div>
    <p class="hero-sub">
        Sistem klasifikasi pemutihan terumbu karang berbasis <i>Convolutional Neural Network</i> dengan visualisasi Grad-CAM untuk interpretasi keputusan model secara visual.
    </p>
    <div class="badge-row">
        <span class="badge">🪸 NOAA-PIFSC Dataset</span>
        <span class="badge">🌊 Coralscapes Dataset</span>
        <span class="badge">🧠 Custom CNN</span>
        <span class="badge">🔍 Grad-CAM XAI</span>
        <span class="badge">📊 Accuracy 82%</span>
    </div>
</div>
""", unsafe_allow_html=True)

# METRICS
st.markdown("""
<div class="metric-grid">
    <div class="metric-card"><div class="metric-val">82%</div><div class="metric-lbl">Accuracy</div></div>
    <div class="metric-card"><div class="metric-val">0.82</div><div class="metric-lbl">Macro F1-Score</div></div>
    <div class="metric-card"><div class="metric-val">0.82</div><div class="metric-lbl">Recall Bleached</div></div>
    <div class="metric-card"><div class="metric-val">8.2K</div><div class="metric-lbl">Total Data Training</div></div>
</div>
""", unsafe_allow_html=True)

# LOAD MODEL
try:
    model = load_cnn_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ── DEBUG SEMENTARA ──
st.write("Layer names:", [l.name for l in model.layers])
dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
dummy_prob = float(model.predict(dummy, verbose=0)[0][0])
st.write(f"Dummy prob (all zeros): {dummy_prob:.6f}")
ones = np.ones((1, 128, 128, 3), dtype=np.float32) * 255
ones_prob = float(model.predict(ones, verbose=0)[0][0])
st.write(f"Dummy prob (all 255): {ones_prob:.6f}")
# ── END DEBUG ──

# ── SECTION 1: SAMPLE ──
st.markdown('<div class="section-label">Contoh Prediksi</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Sampel dari Data Test</div>', unsafe_allow_html=True)

samples_exist = any(os.path.exists(s["path"]) for s in SAMPLE_IMAGES)

if samples_exist:
    for i in range(0, len(SAMPLE_IMAGES), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(SAMPLE_IMAGES):
                break
            sample = SAMPLE_IMAGES[idx]
            if not os.path.exists(sample["path"]):
                continue
            with col:
                img_pil = Image.open(sample["path"]).convert("RGB")
                img_rgb = np.array(img_pil)
                with st.spinner(f"Memproses sampel {idx+1}..."):
                    pred_label, pred_prob, bleach_pct, img, heatmap, superimposed = predict_and_visualize(img_rgb, model)
                st.markdown(f"**Sampel {idx+1}** — Ground Truth: `{sample['label']}` · Sumber: `{sample['source']}`")
                show_result(pred_label, pred_prob, bleach_pct, img, heatmap, superimposed)
                st.markdown("<hr class='divider'>", unsafe_allow_html=True)
else:
    st.info("📁 Letakkan gambar sampel di folder `samples/` dalam repo.")

# ── SECTION 2: UPLOAD ──
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Coba Sendiri</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload Gambar Karang</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload gambar terumbu karang (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown('<p class="upload-hint">Mendukung format JPG dan PNG. Gambar akan di-resize ke 128×128 piksel secara otomatis.</p>', unsafe_allow_html=True)

if uploaded is not None:
    img_pil = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(img_pil)
    with st.spinner("Menganalisis gambar..."):
        pred_label, pred_prob, bleach_pct, img, heatmap, superimposed = predict_and_visualize(img_rgb, model)
    show_result(pred_label, pred_prob, bleach_pct, img, heatmap, superimposed)

# ── SECTION 3: ABOUT ──
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Tentang Penelitian</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Metodologi</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="info-box">
    <b>Dataset</b><br>
    Penelitian ini menggunakan dua dataset: <b>NOAA-PIFSC-ESD Coral Bleaching Dataset</b> dari HuggingFace sebagai data utama (kelas Healthy & Bleached), dan <b>Coralscapes Dataset</b> sebagai data tambahan melalui teknik crop berbasis segmentation mask untuk memperkaya variasi data training.
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="info-box">
    <b>Arsitektur Model</b><br>
    Custom CNN 3-blok tanpa transfer learning, terdiri dari Conv2D → MaxPooling per blok, diikuti Dense layer dengan Dropout 0.5 dan L2 regularization. Visualisasi Grad-CAM digunakan untuk menginterpretasikan area fokus model dalam pengambilan keputusan klasifikasi.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<br>
<div style="text-align:center; color:#2d4a35; font-size:0.78rem; padding-bottom:1rem;">
    CoralSense · Penulisan Ilmiah · Teknik Informatika · 2026
</div>
""", unsafe_allow_html=True)
