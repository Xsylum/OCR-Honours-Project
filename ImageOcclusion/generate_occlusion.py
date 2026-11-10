import cv2
import numpy as np
import os

def add_random_obstruction(img, n_shapes=1, min_size_frac=0.25, max_size_frac=0.9,
                           opacity=1.0, color=(30, 30, 30), shape="ellipse",
                           shadow=True, shadow_strength=0.9, shadow_blur_frac=0.03,
                           shadow_offset_frac=(0.0, 0.0), seed=None):
    """
    img: BGR image (numpy array)
    n_shapes: how many obstructions to draw
    min_size_frac / max_size_frac: shape size as a fraction of image dimensions
    opacity: 1.0 = fully opaque, <1.0 = semi-transparent
    color: BGR tuple
    shadow: draw a soft shadow under the shape
    shadow_strength: 0-1, how dark the shadow gets at its center
    shadow_blur_frac: blur spread as a fraction of the larger image dimension
    shadow_offset_frac: (dx, dy) shadow shift as a fraction of (w, h); (0, 0) gives an even halo
    """
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    result = img.copy()

    min_w = max(2, int(w * min_size_frac))
    max_w = max(min_w + 1, int(w * max_size_frac))
    min_h = max(2, int(h * min_size_frac))
    max_h = max(min_h + 1, int(h * max_size_frac))

    dx = int(w * shadow_offset_frac[0])
    dy = int(h * shadow_offset_frac[1])
    sigma = max(h, w) * shadow_blur_frac

    for _ in range(n_shapes):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        sw = int(rng.integers(min_w, max_w))
        sh = int(rng.integers(min_h, max_h))
        angle = int(rng.integers(0, 180))
        axes = (sw // 2, sh // 2)

        if shape != "ellipse":
            # not sure if it's worth having the shape as parameter or if it doesn't matter
            continue

        if shadow:
            # draw the (offset) ellipse on a float mask, then blur it
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, (cx + dx, cy + dy), axes, angle, 0, 360, 1.0, thickness=-1)
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)

            # darken the image where the mask is
            alpha = (mask * shadow_strength)[..., None]
            result = (result * (1 - alpha)).astype(np.uint8)

        # draw the actual ellipse on top
        overlay = result.copy()
        cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, color, thickness=-1)
        result = cv2.addWeighted(overlay, opacity, result, 1 - opacity, 0)

    return result