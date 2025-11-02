import cv2
import numpy as np
import os

def cartoonize_image(input_path, output_path):
    img = cv2.imread(input_path)
    img = cv2.resize(img, (800, 800))

    # Step 1: Base smoothing
    base = cv2.edgePreservingFilter(img, flags=2, sigma_s=100, sigma_r=0.3)
    base = cv2.bilateralFilter(base, d=9, sigmaColor=60, sigmaSpace=90)

    # Step 2: Adaptive color boost
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.2, 0, 255).astype(np.uint8)
    v = cv2.equalizeHist(v)
    color_enhanced = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    # Step 3: Color quantization
    Z = color_enhanced.reshape((-1, 3))
    Z = np.float32(Z)
    K = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(color_enhanced.shape)

    # Step 4: Blend quantized + enhanced color
    blend_ratio = 0.25
    anime_tone_corrected = cv2.addWeighted(quantized.astype(np.float32), 1 - blend_ratio,
                                           color_enhanced.astype(np.float32), blend_ratio, 0)
    anime_tone_corrected = np.clip(anime_tone_corrected, 0, 255).astype(np.uint8)

    # Step 5: Edge detection
    gray = cv2.cvtColor(color_enhanced, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    edges_inv = cv2.bitwise_not(edges)
    edges_color = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

    # Step 6: Merge edges and tones
    anime_final = cv2.bitwise_and(anime_tone_corrected, edges_color)

    # Step 7: Sharpen
    sharpened = cv2.addWeighted(anime_final, 1.15, cv2.GaussianBlur(anime_final, (3, 3), 0), -0.15, 0)
    anime_final = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Save output
    cv2.imwrite(output_path, anime_final)
    return output_path
