"""3D occluders rendered with the same pinhole camera as the label.

World frame matches camera.py: label plane at z = 0 (mm), camera on the -z side, so "above
the paper" means negative z. Because occluders are real 3D geometry, they hide different
text from different viewpoints.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

NEAR_MM = 3.0  # geometry closer to the camera than this is clipped
SUBPIX = 4  # cv2 drawing fixed-point shift (1/16 px precision)


def project_points(cam, P):
    """World points (n, 3) -> pixel coords (n, 2) and camera-space depth (n,)."""
    R, t, _ = cam.pose()
    Pc = np.asarray(P, np.float64) @ R.T + t
    uv = Pc @ cam.K.T
    return uv[:, :2] / uv[:, 2:3], Pc[:, 2]


def clip_segment(cam, A, B):
    """Clip a 3D segment to the part at least NEAR_MM in front of the camera (None if none is)."""
    _, (za, zb) = project_points(cam, [A, B])
    if za < NEAR_MM and zb < NEAR_MM:
        return None
    A, B = np.asarray(A, float), np.asarray(B, float)
    if za < NEAR_MM:
        A = A + (B - A) * (NEAR_MM - za) / (zb - za)
    elif zb < NEAR_MM:
        B = B + (A - B) * (NEAR_MM - zb) / (za - zb)
    return A, B


def rod_polygon(cam, A, B, diameter_mm, width_scale=1.0, offset=0.0):
    """Image polygon of a thin rod from A to B whose width follows perspective (None if invisible).

    `offset` shifts the strip sideways (in units of the local half-width), used for highlights.
    """
    seg = clip_segment(cam, A, B)
    if seg is None:
        return None
    (a, b), (za, zb) = project_points(cam, seg)
    d = b - a
    length = np.hypot(*d)
    if length < 1e-6:
        return None
    perp = np.array([-d[1], d[0]]) / length
    f = cam.K[0, 0]
    ha, hb = f * diameter_mm / 2 / za, f * diameter_mm / 2 / zb
    ca, cb = a + perp * ha * offset, b + perp * hb * offset
    ha, hb = ha * width_scale, hb * width_scale
    return np.array([ca + perp * ha, cb + perp * hb, cb - perp * hb, ca - perp * ha])


def _fill(img, poly, color):
    cv2.fillPoly(img, [np.round(poly * (1 << SUBPIX)).astype(np.int32)], color, cv2.LINE_AA, SUBPIX)


def _disc(img, centre, radius, color):
    c = tuple(int(round(v * (1 << SUBPIX))) for v in centre)
    cv2.circle(img, c, max(1, int(round(radius * (1 << SUBPIX)))), color, -1, cv2.LINE_AA, SUBPIX)


@dataclass
class Pin:
    """An insect pin through the label: shaft from below the paper up to a ball head."""
    pierce_mm: tuple  # where the shaft goes through the label (label centre = 0, 0)
    length_mm: float = 12.0  # shaft length above the paper, up to the head
    below_mm: float = 4.0  # shaft visible below the paper
    diameter_mm: float = 0.45
    head_diameter_mm: float = 1.3
    shaft_gray: int = 150
    head_gray: int = 25
    lean_mm: tuple = (0.0, 0.0)  # sideways offset of the head (pins are rarely perfectly vertical)

    def points(self):
        px, py = self.pierce_mm
        pierce = np.array([px, py, 0.0])
        lean = np.array([*self.lean_mm, 0.0])
        top = pierce + np.array([0, 0, -self.length_mm]) + lean
        bottom = pierce + np.array([0, 0, self.below_mm]) - lean * self.below_mm / self.length_mm
        return pierce, top, bottom

    def to_json(self):
        return {"type": "pin", **{k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}}


def sample_pin(rng, label_mm):
    """Random pin through the label (anywhere but the outer 12%), numpy Generator `rng`."""
    w, h = label_mm
    return Pin(pierce_mm=(float(rng.uniform(-0.38, 0.38) * w), float(rng.uniform(-0.38, 0.38) * h)),
               length_mm=float(rng.uniform(5, 15)),
               below_mm=float(rng.uniform(2, 6)),
               diameter_mm=float(rng.uniform(0.35, 0.55)),
               head_diameter_mm=float(rng.uniform(1.0, 1.8)),
               shaft_gray=int(rng.uniform(120, 175)),
               head_gray=int(rng.choice([20, 35, 60, 200])),
               lean_mm=(float(rng.normal(0, 0.6)), float(rng.normal(0, 0.6))))


def pin_shadow(cam, pin, light_dir, size):
    """Soft shadow mask (float 0-1) the pin casts on the label plane under a directional light."""
    lx, ly, lz = light_dir
    pierce, top, _ = pin.points()
    s = -top[2] / lz  # walk from the head along the light ray down to z = 0
    tip = top + s * np.array([lx, ly, lz])
    mask = np.zeros(size[::-1], np.uint8)
    poly = rod_polygon(cam, pierce, tip, pin.diameter_mm * 1.3)
    if poly is not None:
        _fill(mask, poly, 255)
    (c,), (z,) = project_points(cam, [tip])
    if z > NEAR_MM:
        _disc(mask, c, cam.K[0, 0] * pin.head_diameter_mm / 2 / z, 255)
    _, (zp,) = project_points(cam, [pierce])
    penumbra_px = float(np.clip(cam.K[0, 0] * 0.3 / max(zp, NEAR_MM), 1.0, 8.0))  # ~0.3 mm soft edge
    return cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), penumbra_px)


def draw_pin(photo, cam, pin, label_alpha, light_dir=None, shadow_strength=0.35):
    """Draw the pin (and its shadow) into a uint8 BGR photo in place.

    Returns a bool mask of pixels where the pin is in front of the label (the occlusion mask).
    """
    h, w = photo.shape[:2]
    pierce, top, bottom = pin.points()

    if light_dir is not None:
        shade = 1 - shadow_strength * pin_shadow(cam, pin, light_dir, (w, h)) * label_alpha
        photo[:] = np.clip(photo * shade[..., None], 0, 255).astype(np.uint8)

    # Shaft below the paper: only visible where the label isn't covering it.
    below = rod_polygon(cam, pierce, bottom, pin.diameter_mm)
    if below is not None:
        layer = photo.copy()
        _fill(layer, below, (pin.shaft_gray - 30,) * 3)
        hidden = label_alpha[..., None] > 0.5
        photo[:] = np.where(hidden, photo, layer)

    occ = np.zeros((h, w), np.uint8)
    shaft = rod_polygon(cam, pierce, top, pin.diameter_mm)
    if shaft is not None:
        _fill(photo, shaft, (pin.shaft_gray,) * 3)
        _fill(occ, shaft, 255)
        highlight = rod_polygon(cam, pierce, top, pin.diameter_mm, width_scale=0.3, offset=0.35)
        if highlight is not None:
            _fill(photo, highlight, (min(255, pin.shaft_gray + 80),) * 3)

    (c,), (z,) = project_points(cam, [top])
    if z > NEAR_MM:
        r = cam.K[0, 0] * pin.head_diameter_mm / 2 / z
        _disc(photo, c, r, (pin.head_gray,) * 3)
        _disc(occ, c, r, 255)
        _disc(photo, c - np.array([0.35, 0.35]) * r, 0.25 * r, (min(255, pin.head_gray + 120),) * 3)
    return occ > 127
