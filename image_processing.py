import cv2
import numpy as np


def ubah_ke_grayscale(image_np):
    """Mengubah citra RGB/BGR menjadi Grayscale."""
    return cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)


def terapkan_threshold(image_np, threshold_val=128):
    """Menerapkan thresholding biner sederhana."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    _, biner = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    return biner


def terapkan_gaussian_blur(image_np, kernel_size=5):
    """Menerapkan penapisan Gaussian melembutkan noise."""
    # Ukuran kernel Gaussian harus ganjil
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)
