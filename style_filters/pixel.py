import cv2
import numpy as np
import os

def pixelate_image(input_path, output_path):
    # Step 1: Read and resize
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("⚠️ Image not found or invalid path!")
    img = cv2.resize(img, (800, 800))

    # Step 2: Automatically set pixelation scale based on width
    if img.shape[1] > 2000:
        scale = 0.05     # large image
    elif img.shape[1] > 1000:
        scale = 0.1      # medium image
    else:
        scale = 0.2      # small image

    # Step 3: Apply pixelation (shrink + enlarge)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    pixel_art = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Step 4: Optional enhancement – slightly sharpen pixels for clarity
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    pixel_art = cv2.filter2D(pixel_art, -1, kernel)

    # Step 5: Save final pixelated image
    cv2.imwrite(output_path, pixel_art)
    return output_path
