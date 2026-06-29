import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os

# ─────────────────────────────────────────
# GLOBAL PAGE CONFIG & STYLING
# ─────────────────────────────────────────
st.set_page_config(
    page_title="CoralSense — Deteksi Pemutihan Karang",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Terpadu untuk Kedua Halaman
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
html, body, [data-testid="stAppViewContainer"] { font-family: 'DM Sans', sans-serif; background: #0a0f0d; color: #e8ede9; }
[data-testid="stSidebar"] { background-color: #050a07 !important; border-right: 1px solid #142e1f; }

.hero-wrapper {
    background: linear-gradient(135deg, #0d2018 0%, #0a1a12 40%, #071510 100%);
    border: 1px solid #1a3d28; border-radius: 16px;
    padding: 2.5rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800; color: #e8ede9; line-height: 1.1; margin: 0 0 0.5rem 0; letter-spacing: -1px; }
.hero-title span { color: #20b464; }
.hero-sub { font-size: 1.05rem; color: #8aab92; font-weight: 300; margin: 0 0 1.5rem 0; max-width: 650px; line-height: 1.6; }

.badge-row { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.badge { background: rgba(32,180,100,0.08); border: 1px solid rgba(32,180,100,0.25); color: #20b464; padding: 0.3rem 0.85rem; border-radius: 999px; font-size: 0.75rem; font-weight: 500; }

.section-label { font-family: 'Syne', sans-serif; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: #20b464; margin-bottom: 0.3rem; }
.section-title { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; color: #e8ede9; margin: 0 0 1.5rem 0; }

.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
.metric-card { background: #0d1a12; border: 1px solid #1a3d28; border-radius: 12px; padding: 1.2rem; text-align: center; transition: all 0.3s ease; }
.metric-card:hover { border-color: #20b464; transform: translateY(-2px); }
.metric-val { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; color: #20b464; line-height: 1; margin-bottom: 0.4rem; }
.metric-lbl { font-size: 0.8rem; color: #6b8c74; font-weight: 400; }

.pred-box { border-radius: 12px; padding: 1.2rem; text-align: center; margin-top: 1.5rem; }
.pred-bleached { background: linear-gradient(135deg, #2d0f0f, #1a0a0a); border: 1px solid #7f1d1d; }
.pred-healthy { background: linear-gradient(135deg, #0d2018, #071510); border: 1px solid #1a3d28; }
.pred-label { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; margin-bottom: 0.3rem; }
.pred-prob { font-size: 0.88rem; color: #a3c2ab; font-weight: 300; }

.info-box { background: #09140e; border-left: 4px solid #20b464; border-radius: 0 12px 12px 0; padding: 1.2rem; margin-top: 1rem; font-size: 0.9rem; color: #8aab92; line-height: 1.6; border-top: 1px solid #142e1f; border-right: 1px solid #142e1f; border-bottom: 1px solid #142e1f;}
.divider { border: none; border-top: 1px solid #142e1f; margin: 2rem 0; }
.upload-hint { font-size: 0.82rem; color: #52755c; margin-top: 0.6rem; font-style: italic; }

[data-testid="stFileUploader"] { background: #07120c; border: 2px dashed #1a3d28; border-radius: 12px; padding: 1rem; }
[data-testid="stFileUploader"] section { padding: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CORE CONFIG & CORE FUNCTIONS
# ─────────────────────────────────────────
IMG_SIZE = (128, 128)
MODEL_PATH = "dashboard/cnn_coral_bleaching.keras"

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

def preprocess_image(img_array):
    img_pil = Image.fromarray(img_array).resize(IMG_SIZE, Image.BILINEAR)
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
    maxc, minc = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    s = np.where(maxc != 0, (maxc - minc) / maxc, 0)
    v = maxc
    mask = (s < (60/255.0)) & (v > (170/255.0))
    return (np.sum(mask) / mask.size) * 100

def predict_and_visualize(img_rgb, model):
    img = preprocess_image(img_rgb)
    img_tensor = tf.cast(np.expand_dims(img, axis=0), tf.float32)

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
        st.image(img, caption="Gambar Input (128x128)", use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor('#0a0f0d')
        ax.imshow(heatmap, cmap='jet')
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.caption("Grad-CAM Activation Heatmap")
    with col3:
        st.image(superimposed, caption="Superimposed Region Analysis", use_container_width=True)

    if pred_label == "Bleached":
        st.markdown(f"""
        <div class="pred-box pred-bleached">
            <div class="pred-label" style="color:#f87171;">🔴 {pred_label}</div>
            <div class="pred-prob">Probabilitas Bleaching: <b>{pred_prob:.4f}</b> &nbsp;|&nbsp; Estimasi Area Putih: <b>{bleach_pct:.1f}%</b></div>
        </div>
        <div class="info-box" style="border-left-color: #f87171;">
            ⚠️ <b>Karang terdeteksi mengalami pemutihan (bleaching).</b> Terumbu karang kehilangan alga simbiotik <i>zooxanthellae</i> akibat stres lingkungan (suhu ekstrem). Sinyal visual merah pada heatmap menunjukkan region dominan putih/pudar yang memicu keputusan klasifikasi model.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="pred-box pred-healthy">
            <div class="pred-label" style="color:#4ade80;">🟢 {pred_label}</div>
            <div class="pred-prob">Probabilitas Bleaching: <b>{pred_prob:.4f}</b> &nbsp;|&nbsp; Estimasi Area Putih: <b>{bleach_pct:.1f}%</b></div>
        </div>
        <div class="info-box">
            ✅ <b>Terumbu karang terdeteksi sehat.</b> Warna yang heterogen dan solid menandakan populasi alga simbiotik melimpah. Model berfokus pada variasi spektrum warna alami dan struktur tekstur karang yang solid.
        </div>
        """, unsafe_allow_html=True)

# LOAD MODEL GLOBAL INITIALIZATION
try:
    model = load_cnn_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


# ─────────────────────────────────────────
# PAGE 1: UPLOAD & PREDICTION VIEW
# ─────────────────────────────────────────
def page_predict():
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-title">Coral<span>Sense</span> Analisis</div>
        <p class="hero-sub">
            Unggah citra terumbu karang Anda untuk mendeteksi tingkat pemutihan secara real-time menggunakan arsitektur deep learning kustom yang didukung akuntabilitas Grad-CAM.
        </p>
        <div class="badge-row">
            <span class="badge">🌐 Real-Time Inference</span>
            <span class="badge">🧠 Custom CNN Loaded</span>
            <span class="badge">🔍 Grad-CAM Region Mapping</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Mesin Prediksi</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Gambar Karang</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload gambar terumbu karang (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown('<p class="upload-hint">Mendukung file berkstensi .jpg, .jpeg, dan .png. Gambar otomatis diproses ke dimensi target 128×128 piksel.</p>', unsafe_allow_html=True)

    if uploaded is not None:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        img_pil = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(img_pil)
        with st.spinner("Mengekstrak fitur citra dan menjalankan konvolusi..."):
            pred_label, pred_prob, bleach_pct, img, heatmap, superimposed = predict_and_visualize(img_rgb, model)
        show_result(pred_label, pred_prob, bleach_pct, img, heatmap, superimposed)


# ─────────────────────────────────────────
# PAGE 2: METRICS & SAMPLE VIEW
# ─────────────────────────────────────────
def page_metrics():
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-title">Evaluasi <span>& Performa</span></div>
        <p class="hero-sub">
            Eksplorasi metrik pengujian model hasil latih, metodologi komposit dataset, serta simulasi pengujian pada representasi data validasi.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # METRICS GRID
    st.markdown('<div class="section-label">Performa Klasifikasi</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Metrik Evaluasi Model</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-grid">
        <div class="metric-card"><div class="metric-val">82.0%</div><div class="metric-lbl">Akurasi Validasi</div></div>
        <div class="metric-card"><div class="metric-val">0.82</div><div class="metric-lbl">Macro F1-Score</div></div>
        <div class="metric-card"><div class="metric-val">0.82</div><div class="metric-lbl">Recall Bleached</div></div>
        <div class="metric-card"><div class="metric-val">8.2K</div><div class="metric-lbl">Total Citra Training</div></div>
    </div>
    """, unsafe_allow_html=True)

    # METHODOLOGY
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Arsitektur & Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Metodologi Riset</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
        <b>Pipeline Komposit Dataset</b><br>
        Menggabungkan representasi data dari <b>NOAA-PIFSC-ESD Coral Bleaching Dataset</b> (sebagai jangkar utama kelas kondisi) dan <b>Coralscapes Dataset</b> yang diperkaya lewat teknik cropping berbasis segmentation mask demi mengisolasi region terumbu secara presisi dari noise latar belakang air laut.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-box">
        <b>Arsitektur CNN Kustom</b><br>
        Arsitektur dibangun dari dasar tanpa transfer learning, menerapkan 3-blok interkoneksi konvolusi (Conv2D → MaxPool2D) dengan penambahan regularisasi L2 serta Dropout 0.5 pada fully-connected dense layer. Grad-CAM disuntikkan pada lapisan konvolusi terakhir untuk menjamin akuntabilitas prediksi.
        </div>
        """, unsafe_allow_html=True)

    # SAMPLES SCRIPT
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Pengujian Sampel</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Eksperimen Citra Test</div>', unsafe_allow_html=True)

    SAMPLE_IMAGES = [
        {"path": "dashboard/samples/noaa_healthy.jpg",  "label": "Healthy",  "source": "NOAA-PIFSC"},
        {"path": "dashboard/samples/cs_healthy.jpg",    "label": "Healthy",  "source": "Coralscapes"},
        {"path": "dashboard/samples/noaa_bleached.jpg", "label": "Bleached", "source": "NOAA-PIFSC"},
        {"path": "dashboard/samples/cs_bleached.jpg",   "label": "Bleached", "source": "Coralscapes"},
    ]

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
                    pred_label, pred_prob, bleach_pct, img, heatmap, superimposed = predict_and_visualize(img_rgb, model)
                    st.markdown(f"📊 **Sampel {idx+1}** — Ground Truth: `{sample['label']}` · Source: `{sample['source']}`")
                    show_result(pred_label, pred_prob, bleach_pct, img, heatmap, superimposed)
                    st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("📁 Folder citra sampel tidak ditemukan. Letakkan file gambar pada direktori `dashboard/samples/` untuk memuat visualisasi otomatis.")


# ─────────────────────────────────────────
# STREAMLIT NAVIGATION ROUTER
# ─────────────────────────────────────────

# Judul sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0rem 0.5rem 0rem;">
        <h2 style="font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; color: #e8ede9; margin: 0; letter-spacing: -0.5px;">
            Coral<span style="color: #20b464;">Sense</span>
        </h2>
        <p style="font-size: 0.72rem; color: #6b8c74; margin: 0.2rem 0 0 0; text-transform: uppercase; letter-spacing: 0.12em;">
            Navigation Menu
        </p>
    </div>
    <hr style="border: none; border-top: 1px solid #142e1f; margin-top: 0.5rem; margin-bottom: 0.5rem;">
    """, unsafe_allow_html=True)

# Navigasi halaman
pg = st.navigation([
    st.Page(page_metrics, title="Analisis Citra", icon="📊"),
    st.Page(page_predict, title="Upload & Prediksi Kesehatan Karang", icon="🪸")
])

# Jalankan router halaman
pg.run()

# FOOTER CREDITS GLOBAL
st.markdown("""
<div style="text-align:center; color:#2d4a35; font-size:0.75rem; padding-top:2rem; padding-bottom:1rem; border-top: 1px solid #142e1f;">
    CoralSense — Penulisan Ilmiah · Teknik Informatika · Universitas Gunadarma · 2026
</div>
""", unsafe_allow_html=True)