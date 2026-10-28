import cv2
import numpy as np

img1 = cv2.imread('result_cropped.png', cv2.IMREAD_UNCHANGED)
img2 = cv2.imread('result_cropped.png', cv2.IMREAD_UNCHANGED)
spacer_width = 5 # number of pixels between the syllables... tune until it looks decent

if img1.shape != img2.shape:
    # Ensure you are referencing shape[0] (height) or shape[1] (width) for the math
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    new_width = int(w2 * (h1 / h2))
    img2 = cv2.resize(img2, (new_width, h1))

height, _, channels = img1.shape
spacer = np.ones((height, spacer_width, channels), dtype=np.uint8)

# Attach side-by-side
horizontal_combined = cv2.hconcat([img1, spacer, img2])

cv2.imwrite('horizontal_output.png', horizontal_combined)