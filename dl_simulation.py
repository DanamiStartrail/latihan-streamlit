import numpy as np


def simulasi_model_b(image_pil, nama_file=""):
    """Menghasilkan contoh keluaran simulasi untuk Model B (MobileNetV2)
    sebelum model deep learning asli tersedia.
    """
    # Contoh 10 kelas makanan Indonesia sesuai rencana komparasi proyek
    daftar_kelas = [
        "Bakso",
        "Gado-gado",
        "Gudeg",
        "Nasi Goreng",
        "Nasi Padang",
        "Pempek",
        "Rawon",
        "Rendang",
        "Sate",
        "Soto",
    ]

    # Gunakan ukuran/nilai piksel gambar agar hasil simulasi deterministik per citra
    arr = np.asarray(image_pil.convert("RGB").resize((224, 224)))
    seed_val = int(arr.mean() * 100 + arr.shape[0]) % len(daftar_kelas)

    # Buat distribusi probabilitas contoh yang jumlahnya 1.0
    rng = np.random.default_rng(seed_val)
    raw_scores = rng.dirichlet(np.ones(len(daftar_kelas)) * 0.6)
    top_idx = int(np.argmax(raw_scores))

    top_3 = [
        {
            "Kelas": daftar_kelas[i],
            "Probabilitas (Simulasi)": round(float(raw_scores[i]), 4),
        }
        for i in np.argsort(raw_scores)[::-1][:3]
    ]

    return {
        "status": "SIMULASI (Placeholder - Belum Menggunakan Bobot MobileNetV2 Asli)",
        "input_resolution": "224x224",
        "prediksi_utama": daftar_kelas[top_idx],
        "confidence": round(float(raw_scores[top_idx]), 4),
        "top_3": top_3,
        "catatan": (
            f"Simulasi untuk '{nama_file or 'citra unggahan'}'. "
            "Nilai probabilitas ini hanya contoh tata letak UI dan bukan hasil inferensi model final."
        ),
    }
