import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops


def proses_pcd_model_a(image_pil):
    """Mengimplementasikan 4 Tahap Model A PCD Klasik dari Proposal Tubes:

    1. Resize (256x256) & Gaussian Blur
    2. Konversi HSV & Otsu Thresholding pada Kanal S (Saturasi)
    3. Morphological Opening & Closing
    4. Ekstraksi Fitur Histogram HSV & 4 Fitur GLCM (pada area masker)
    """
    # Konversi PIL Image ke NumPy Array (RGB)
    img_rgb = np.array(image_pil.convert("RGB"))

    # Tahap 1: Resizing ke 256x256 & Gaussian Blur (5x5)
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)

    # Tahap 2: Transformasi RGB ke HSV & Otsu pada Kanal S
    hsv = cv2.cvtColor(img_blurred, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1]  # Kanal Saturasi
    _, mask_otsu = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tahap 3: Operasi Morfologi (Opening diikuti Closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_open = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel)
    mask_final = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    # Terapkan masker ke citra hasil segmentasi makanan
    segmented_food = cv2.bitwise_and(img_resized, img_resized, mask=mask_final)

    # Tahap 4A: Ekstraksi Fitur Histogram HSV (8 bin per kanal) pada area masker
    # Parameter: [images], [channel], mask, [histSize], [ranges]
    hist_h = cv2.calcHist([hsv], [0], mask_final, [8], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], mask_final, [8], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], mask_final, [8], [0, 256])

    # Normalisasi histogram agar total sum = 1
    hist_features = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    if hist_features.sum() > 0:
        hist_features = hist_features / hist_features.sum()

    # Tahap 4B: Ekstraksi 4 Fitur Tekstur GLCM pada Grayscale ber-masker
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask_final)

    # Hitung GLCM (jarak 1 piksel, sudut 0 derajat)
    glcm = graycomatrix(
        gray_masked,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )

    glcm_contrast = graycoprops(glcm, "contrast")[0, 0]
    glcm_homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    glcm_energy = graycoprops(glcm, "energy")[0, 0]
    glcm_correlation = graycoprops(glcm, "correlation")[0, 0]

    glcm_features = {
        "Contrast": round(float(glcm_contrast), 4),
        "Homogeneity": round(float(glcm_homogeneity), 4),
        "Energy": round(float(glcm_energy), 4),
        "Correlation": round(float(glcm_correlation), 4),
    }

    # Mengembalikan semua tahapan perantara untuk visualisasi Streamlit
    return {
        "resized": img_resized,
        "blurred": img_blurred,
        "s_channel": s_channel,
        "mask_otsu": mask_otsu,
        "mask_final": mask_final,
        "segmented": segmented_food,
        "hist_features": hist_features,
        "glcm_features": glcm_features,
    }
