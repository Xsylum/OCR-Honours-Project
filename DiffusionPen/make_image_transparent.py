# Source - https://stackoverflow.com/a/59443225
# Posted by Mark Setchell, modified by community. See post 'Timeline' for change history
# Retrieved 2026-10-19, License - CC BY-SA 4.0

import cv2
import numpy as np

# Load Image in grayscale
im = cv2.imread('image_samples\\prox_style_1.png', cv2.IMREAD_GRAYSCALE)

# make near-white pixels transparent
threshold = 220
alpha = np.where(im >= threshold, 0, 255).astype(np.uint8)

# Create RGBA image
result = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
result[:, :, 3] = alpha

# Save as PNG
cv2.imwrite('result.png', result)