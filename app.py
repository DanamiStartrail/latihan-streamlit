import streamlit as st
from PIL import Image

from pcd_pipeline import proses_pcd_model_a

# Pengaturan Halaman
st.set_page_config(page_title="PCD NMAX - Pengenalan Makanan", layout="wide")

st.title("🍲 Demo Pipeline PCD Klasik (Model A) - Kelompok NMAX")
st.write(
    "Uji coba visualisasi 4 tahap pengolahan citra digital untuk segmentasi dan ekstraksi fitur makanan."
)

# Komponen Pengunggah File
uploaded_file = st.sidebar.file_uploader(
    "Pilih Foto Makanan...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)

    # Eksekusi Pipeline PCD
    with st.spinner("Memproses tahapan PCD..."):
        res = proses_pcd_model_a(image_pil)

    st.subheader("1. Visualisasi Tahapan Pengolahan Citra Digital")

    # Baris 1: Citra Asli, Gaussian Blur, Kanal S (HSV)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(
            res["resized"], caption="1A. Resized (256x256)", use_container_width=True
        )
    with col2:
        st.image(
            res["blurred"], caption="1B. Gaussian Blur (5x5)", use_container_width=True
        )
    with col3:
        st.image(
            res["s_channel"],
            caption="2A. Kanal Saturasi (S) HSV",
            use_container_width=True,
            clamp=True,
        )

    # Baris 2: Threshold Otsu, Masker Morfologi, Hasil Segmentasi
    col4, col5, col6 = st.columns(3)
    with col4:
        st.image(
            res["mask_otsu"], caption="2B. Otsu Thresholding", use_container_width=True
        )
    with col5:
        st.image(
            res["mask_final"],
            caption="3. Masker Morfologi (Open+Close)",
            use_container_width=True,
        )
    with col6:
        st.image(
            res["segmented"],
            caption="Hasil Akhir Segmentasi Makanan",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("2. Hasil Ekstraksi Fitur Tekstur (GLCM)")

    # Menampilkan Fitur GLCM dalam bentuk Metric Card
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Contrast", res["glcm_features"]["Contrast"])
    m2.metric("Homogeneity", res["glcm_features"]["Homogeneity"])
    m3.metric("Energy", res["glcm_features"]["Energy"])
    m4.metric("Correlation", res["glcm_features"]["Correlation"])

    st.markdown("---")
    st.subheader("3. Vektor Fitur Histogram Warna HSV")
    st.bar_chart(res["hist_features"])
else:
    st.info(
        "💡 Silakan unggah foto makanan di panel samping kiri untuk memulai uji coba!"
    )
