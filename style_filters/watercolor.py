import cv2
import numpy as np
import os

def watercolor_image(input_path, output_path):
    # Step 1: Read and resize
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("⚠️ Image not found or invalid path!")
    img = cv2.resize(img, (800, 800))
    original = img.copy()

    # Step 2: K-means color quantization
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    K = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(img.shape)

    # Step 3: Bilateral filtering for watercolor smoothness
    watercolor = quantized
    for _ in range(2):
        watercolor = cv2.bilateralFilter(watercolor, d=9, sigmaColor=50, sigmaSpace=50)

    # Step 4: Create soft edges
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    edges = 255 - edges
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)

    # Step 5: Blend soft edges with smoothed watercolor
    watercolor_final = cv2.addWeighted(watercolor, 0.95, edges, 0.05, 0)

    # Step 6: Save output
    cv2.imwrite(output_path, watercolor_final)
    return output_path
