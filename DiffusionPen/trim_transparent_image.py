import cv2
import numpy as np

# Load the RGBA image (including transparency)
im = cv2.imread('result.png', cv2.IMREAD_UNCHANGED)

# Extract alpha channel
alpha = im[:, :, 3]

# Find all non-transparent pixels
coords = cv2.findNonZero((alpha > 0).astype(np.uint8))

if coords is not None:
    # Find bounding rectangle around the content
    x, y, w, h = cv2.boundingRect(coords)

    # Crop to the content
    cropped = im[y:y+h, x:x+w]

    # Save cropped image
    cv2.imwrite('result_cropped.png', cropped)