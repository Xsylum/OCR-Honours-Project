"""Lens distortion (OpenCV / Brown-Conrady model: radial k1, k2, k3 and tangential p1, p2).

The ideal pinhole photo is resampled so that straight lines bend like they do through a real
lens, and the ground-truth points go through the same forward model. The coefficients are
stored in OpenCV order, so `cv2.undistort(photo, K, dist)` recovers the pinhole view.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class Lens:
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    @property
    def coeffs(self):
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

    def to_json(self):
        return {**{k: round(v, 6) for k, v in asdict(self).items()},
                "opencv_dist_coeffs": [round(float(c), 6) for c in self.coeffs]}


def sample_lens(rng, strength=1.0):
    """Mostly barrel distortion (phone / compact cameras), sometimes mild pincushion."""
    return Lens(k1=float(rng.uniform(-0.3, 0.12) * strength), k2=float(rng.uniform(-0.08, 0.08) * strength),
                p1=float(rng.normal(0, 0.0015) * strength), p2=float(rng.normal(0, 0.0015) * strength))


def _max_r2(K, size):
    w, h = size
    return float(max(((x - K[0, 2]) / K[0, 0]) ** 2 + ((y - K[1, 2]) / K[1, 1]) ** 2 for x in (0, w) for y in (0, h)))


def _distort_normalized(lens, x, y, r2_cap):
    r2 = x * x + y * y
    rc = np.minimum(r2, r2_cap)  # beyond the frame, scale linearly so the mapping can't fold back
    radial = 1 + lens.k1 * rc + lens.k2 * rc ** 2 + lens.k3 * rc ** 3
    xd = x * radial + 2 * lens.p1 * x * y + lens.p2 * (r2 + 2 * x * x)
    yd = y * radial + lens.p1 * (r2 + 2 * y * y) + 2 * lens.p2 * x * y
    return xd, yd


def distort_points(lens, K, size, pts):
    """Ideal pinhole pixel coordinates -> where they land in the distorted photo."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    x = (pts[:, 0] - K[0, 2]) / K[0, 0]
    y = (pts[:, 1] - K[1, 2]) / K[1, 1]
    xd, yd = _distort_normalized(lens, x, y, 2 * _max_r2(K, size))
    return np.column_stack([xd * K[0, 0] + K[0, 2], yd * K[1, 1] + K[1, 2]])


def _undistort_normalized(lens, xd, yd, r2_cap, iterations=20):
    x, y = xd.copy(), yd.copy()
    for _ in range(iterations):
        fx, fy = _distort_normalized(lens, x, y, r2_cap)
        x, y = x + (xd - fx), y + (yd - fy)
    return x, y


def distortion_maps(lens, K, size, step=8):
    """remap() maps: for each distorted output pixel, where to sample the ideal image.

    The inverse is solved on a coarse grid and upsampled (the field is smooth)."""
    w, h = size
    gx = np.arange(0, w + step, step, dtype=np.float64)
    gy = np.arange(0, h + step, step, dtype=np.float64)
    xx, yy = np.meshgrid(gx, gy)
    xd, yd = (xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1]
    x, y = _undistort_normalized(lens, xd, yd, 2 * _max_r2(K, size))
    coarse_x = (x * K[0, 0] + K[0, 2]).astype(np.float32)
    coarse_y = (y * K[1, 1] + K[1, 2]).astype(np.float32)
    u, v = np.meshgrid(np.arange(w, dtype=np.float32) / step, np.arange(h, dtype=np.float32) / step)
    return (cv2.remap(coarse_x, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
            cv2.remap(coarse_y, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))


def apply(maps, img, interpolation=cv2.INTER_LINEAR, border=cv2.BORDER_REFLECT):
    return cv2.remap(img, maps[0], maps[1], interpolation, borderMode=border)
