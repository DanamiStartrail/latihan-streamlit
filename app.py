import streamlit as st
from PIL import Image

from dl_simulation import simulasi_model_b
from pcd_pipeline import proses_pcd_model_a

st.set_page_config(
    page_title="Demo Prototipe PCD NMAX (Model A & Simulasi Model B)",
    page_icon="🍲",
    layout="wide",
)

st.title("🍲 Demo Prototipe Pipeline PCD NMAX")
st.write(
    "Prototipe antarmuka untuk uji coba **Model A (PCD Klasik)** dan "
    "**simulasi tampilan Model B (MobileNetV2)** guna mengumpulkan masukan tim."
)

st.warning(
    "⚠️ **Catatan Prototipe:** **Model A** memproses citra secara nyata sesuai tahapan "
    "pengolahan citra klasik, sedangkan **Model B** saat ini masih berupa **simulasi contoh** "
    "untuk keperluan pengujian alur dan tata letak UI sebelum model asli diintegrasikan."
)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Pengaturan Demo")
mode_tampilan = st.sidebar.radio(
    "Pilih Tampilan Mode:",
    [
        "Model A (PCD Klasik)",
        "Model B (Simulasi)",
        "Bandingkan Model A vs Simulasi Model B",
    ],
)

uploaded_file = st.sidebar.file_uploader(
    "Pilih Foto Makanan...", type=["jpg", "jpeg", "png"]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Bagian ini dibuat sebagai contoh hasil demo untuk didiskusikan bersama anggota tim."
)


def render_model_a(res):
    st.subheader("1. Visualisasi Tahapan Model A (PCD Klasik)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(
            res["resized"],
            caption="1A. Resized (256x256)",
            use_container_width=True,
        )
    with col2:
        st.image(
            res["blurred"],
            caption="1B. Gaussian Blur (5x5)",
            use_container_width=True,
        )
    with col3:
        st.image(
            res["s_channel"],
            caption="2A. Kanal Saturasi (S) HSV",
            use_container_width=True,
            clamp=True,
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        st.image(
            res["mask_otsu"],
            caption="2B. Otsu Thresholding",
            use_container_width=True,
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

    st.markdown("#### Fitur Tekstur (GLCM)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Contrast", res["glcm_features"]["Contrast"])
    m2.metric("Homogeneity", res["glcm_features"]["Homogeneity"])
    m3.metric("Energy", res["glcm_features"]["Energy"])
    m4.metric("Correlation", res["glcm_features"]["Correlation"])

    st.markdown("#### Vektor Fitur Histogram Warna HSV")
    st.bar_chart(res["hist_features"])


def render_model_b_simulasi(sim_res):
    st.subheader("2. Simulasi Keluaran Model B (MobileNetV2)")
    st.info(f"Status: **{sim_res['status']}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediksi Contoh", sim_res["prediksi_utama"])
    c2.metric("Skor Contoh", f"{sim_res['confidence'] * 100:.1f}%")
    c3.metric("Resolusi Input Target", sim_res["input_resolution"])

    st.markdown("#### Top-3 Prediksi (Contoh Simulasi)")
    st.dataframe(sim_res["top_3"], use_container_width=True)
    st.caption(sim_res["catatan"])


if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)

    if mode_tampilan == "Model A (PCD Klasik)":
        with st.spinner("Menjalankan Model A..."):
            res_a = proses_pcd_model_a(image_pil)
        render_model_a(res_a)

    elif mode_tampilan == "Model B (Simulasi)":
        sim_b = simulasi_model_b(image_pil, uploaded_file.name)
        render_model_b_simulasi(sim_b)

    else:
        with st.spinner("Menjalankan Model A & menyiapkan simulasi Model B..."):
            res_a = proses_pcd_model_a(image_pil)
            sim_b = simulasi_model_b(image_pil, uploaded_file.name)

        tab_a, tab_b = st.tabs(
            ["Model A (PCD Klasik)", "Model B (Simulasi / Placeholder)"]
        )
        with tab_a:
            render_model_a(res_a)
        with tab_b:
            render_model_b_simulasi(sim_b)
else:
    st.info(
        "💡 Unggah foto makanan di panel samping kiri untuk mencoba "
        "tampilan Model A, simulasi Model B, atau perbandingannya."
    )
