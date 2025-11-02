
import cv2
import numpy as np
import os

def sketch_image(input_path, output_path):
    # Step 1: Read and resize
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("⚠️ Image not found or invalid path!")
    img = cv2.resize(img, (800, 800))

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Enhance tones for better texture
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Step 4: Invert and blur for base sketch tone
    invert = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(invert, (25, 25), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256.0)

    # Step 5: Add subtle paper-like noise
    noise = np.random.normal(0, 6, sketch.shape).astype(np.uint8)
    sketch = cv2.addWeighted(sketch, 0.95, noise, 0.05, 0)

    # Step 6: Generate strong dark outlines
    edges = cv2.Canny(gray, 40, 120)          # more sensitive edge detection
    edges = cv2.dilate(edges, None, iterations=2)  # thicken the edges
    edges = cv2.bitwise_not(edges)

    # Step 7: Blend with stronger outline visibility
    sketch = cv2.addWeighted(sketch, 0.75, edges, 0.35, 0)

    # Step 8: Enhance contrast and darken pencil tones
    sketch = cv2.convertScaleAbs(sketch, alpha=1.4, beta=-20)

    # Step 9: Slight smoothing for realism
    sketch = cv2.GaussianBlur(sketch, (3, 3), 0)

    # Step 10: Save output
    cv2.imwrite(output_path, sketch)
    return output_path
