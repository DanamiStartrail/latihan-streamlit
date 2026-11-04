import cv2
import numpy as np
import streamlit as st

# Impor fungsi pengolahan citra dari modul lokal
from image_processing import (
    terapkan_gaussian_blur,
    terapkan_threshold,
    ubah_ke_grayscale,
)

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="PCD Demo App - Application Engineer",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Aplikasi Pengolahan Citra Digital")
st.write(
    "Selamat datang! Unggah citra untuk mencoba berbagai operasi pengolahan citra."
)

# --- SIDEBAR: PENGATURAN & KONTROL ---
st.sidebar.header("⚙️ Kontrol & Parameter")

# Widget Upload Citra
uploaded_file = st.sidebar.file_uploader(
    "Unggah Citra Masukan", type=["jpg", "png", "jpeg", "tif"]
)

# Widget Pemilihan Operasi
operasi = st.sidebar.selectbox(
    "Pilih Operasi PCD",
    ["Grayscale", "Thresholding (Biner)", "Gaussian Blur (Filtering)"],
)

# Parameter dinamis berdasarkan operasi yang dipilih
if operasi == "Thresholding (Biner)":
    threshold_val = st.sidebar.slider(
        "Nilai Ambang (Threshold)",
        min_value=0,
        max_value=255,
        value=128,
        step=1,
    )
elif operasi == "Gaussian Blur (Filtering)":
    kernel_size = st.sidebar.slider(
        "Ukuran Kernel (Ganjil)", min_value=3, max_value=31, value=5, step=2
    )

# --- AREA UTAMA DISPLAY ---
if uploaded_file is not None:
    # 1. Konversi uploaded file Streamlit ke format NumPy Array (OpenCV BGR)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert BGR ke RGB untuk tampilan awal di Streamlit
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Proses citra berdasarkan pilihan pengguna
    if operasi == "Grayscale":
        img_hasil = ubah_ke_grayscale(img_bgr)
    elif operasi == "Thresholding (Biner)":
        img_hasil = terapkan_threshold(img_bgr, threshold_val)
    elif operasi == "Gaussian Blur (Filtering)":
        img_hasil_bgr = terapkan_gaussian_blur(img_bgr, kernel_size)
        img_hasil = cv2.cvtColor(img_hasil_bgr, cv2.COLOR_BGR2RGB)

    # 3. Tampilkan Citra Side-by-Side (Sebelum & Sesudah)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Citra Asli")
        st.image(
            img_rgb, channels="RGB", use_container_width=True
        )

    with col2:
        st.subheader(f"✨ Hasil Operasi ({operasi})")
        # Menyesuaikan penanganan kanal warna untuk gambar 1-kanal (grayscale/biner) vs 3-kanal (RGB)
        if len(img_hasil.shape) == 2:
            st.image(
                img_hasil, use_container_width=True
            )
        else:
            st.image(
                img_hasil, channels="RGB", use_container_width=True
            )

else:
    st.info("👈 Silakan unggah gambar pada sidebar sebelah kiri untuk memulai.")
