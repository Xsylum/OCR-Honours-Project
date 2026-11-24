"""Clear plastic protective sleeve over the label.

The sleeve is a thin film plane just above the paper (z slightly negative), a bit larger than
the label and slightly misaligned. It is rendered in its own texture space and warped with
the same camera, like the stacked labels:

- haze and tint: the film scatters some light, lifting blacks and lowering text contrast;
- specular glare: a light source reflects off the film. Per texel the reflected light
  direction is compared with the direction to the camera, using the normal of a height field
  with gentle waviness and wrinkles, so highlights follow perspective and break up into
  streaks along creases;
- Fresnel sheen (stronger at grazing angles), a bright cut edge, scratches and dust.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ProjectiveProjection.effects import smooth_noise
from ProjectiveProjection.occluders import plane_homography

PX_PER_MM = 24


@dataclass
class Sleeve:
    z_mm: float = -0.4  # film height above the paper (negative = towards the camera)
    margin_mm: tuple = (2.0, 2.0, 2.0, 2.0)  # left, top, right, bottom overhang past the label
    angle_deg: float = 0.0
    offset_mm: tuple = (0.0, 0.0)
    haze: float = 0.08  # fraction of light scattered by the film
    tint_bgr: tuple = (1.0, 1.0, 1.0)
    gloss: float = 0.8  # peak glare, as a fraction of white
    roughness_deg: float = 6.0  # angular spread of the reflection
    glare_light: tuple = (0.0, 0.0, 1.0)  # direction the reflected light travels (towards +z)
    waviness_mm: float = 0.1
    wrinkles: int = 3
    scratches: int = 10
    scuffs: int = 2  # swirl marks from wiping the film
    fog: float = 0.3  # how strongly fogged / smudged patches blur and whiten what's underneath (0-1)
    fog_blur_mm: float = 0.08  # blur radius of the fogged film, at the paper
    texture_seed: int = 0

    def world_transform(self, label_mm):
        """(A: texture pixel -> world (X, Y, 1) on the film plane, texture size (w, h))."""
        W, H = label_mm
        left, top, right, bottom = self.margin_mm
        m = 1.0 / PX_PER_MM
        a = np.radians(self.angle_deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        A = np.eye(3)
        A[:2, :2] = rot * m
        A[:2, 2] = rot @ np.array([-W / 2 - left, -H / 2 - top]) + np.asarray(self.offset_mm)
        size = (int(round((W + left + right) * PX_PER_MM)), int(round((H + top + bottom) * PX_PER_MM)))
        return A, size

    def to_json(self):
        return {"type": "sleeve", **{k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}}


def sample_sleeve(rng, label_mm, cam):
    """Random sleeve. The glare light is aimed so its mirror reflection lands somewhere around
    the label as seen from this camera (often partly or fully off the label)."""
    W, H = label_mm
    s = Sleeve(z_mm=float(-rng.uniform(0.2, 0.8)),
               margin_mm=tuple(float(m) for m in rng.uniform(0.5, 2.5, 4)),
               angle_deg=float(rng.normal(0, 2.0)),
               offset_mm=(float(rng.normal(0, 0.3)), float(rng.normal(0, 0.3))),
               haze=float(rng.uniform(0.02, 0.08)),
               tint_bgr=(float(rng.uniform(0.96, 1.0)), float(rng.uniform(0.98, 1.0)), float(rng.uniform(0.96, 1.0))),
               gloss=float(rng.uniform(0.3, 0.9)),
               roughness_deg=float(rng.uniform(1.5, 5.0)),
               waviness_mm=float(rng.uniform(0.03, 0.25)),
               wrinkles=int(rng.integers(0, 6)),
               scratches=int(rng.integers(5, 60)),
               scuffs=int(rng.integers(0, 5)),
               fog=float(rng.uniform(0.1, 0.6)),
               fog_blur_mm=float(rng.uniform(0.04, 0.15)),
               texture_seed=int(rng.integers(0, 2 ** 31)))
    _, _, C = cam.pose()
    P = np.array([rng.uniform(-0.9, 0.9) * W, rng.uniform(-1.5, 1.5) * H, s.z_mm])
    v = (C - P) / np.linalg.norm(C - P)
    s.glare_light = tuple(float(c) for c in (v[0], v[1], -v[2]))  # mirror about the film plane
    return s


def _height_field(sleeve, shape, rng):
    """Film height above its plane (mm): smooth waviness plus a few creases."""
    th, tw = shape
    h = smooth_noise(shape, rng, cell=max(8, max(shape) // 3)) * sleeve.waviness_mm
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32) / PX_PER_MM
    for _ in range(sleeve.wrinkles):
        c = np.array([rng.uniform(0, tw), rng.uniform(0, th)]) / PX_PER_MM
        a = rng.uniform(0, np.pi)
        t, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        length, width, amp = rng.uniform(3, 15), rng.uniform(0.2, 0.8), rng.uniform(0.03, 0.15)
        along = (xx - c[0]) * t[0] + (yy - c[1]) * t[1]
        across = (xx - c[0]) * n[0] + (yy - c[1]) * n[1]
        h += amp * np.exp(-(across / width) ** 2) * np.exp(-(along / (length / 2)) ** 4)
    return h


def render_sleeve(sleeve, cam, label_mm):
    """Film appearance in texture space.

    Returns (A, transmission (h, w, 3), added light (h, w, 3) in 0-255, glare (h, w) in 0-1,
    fog (h, w) in 0-1)."""
    A, (tw, th) = sleeve.world_transform(label_mm)
    rng = np.random.default_rng(sleeve.texture_seed)
    hgt = _height_field(sleeve, (th, tw), rng)
    hy, hx = np.gradient(hgt, 1.0 / PX_PER_MM)
    n = np.stack([-hx, -hy, -np.ones_like(hx)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    u, v = np.meshgrid(np.arange(tw, dtype=np.float64), np.arange(th, dtype=np.float64))
    X = A[0, 0] * u + A[0, 1] * v + A[0, 2]
    Y = A[1, 0] * u + A[1, 1] * v + A[1, 2]
    _, _, C = cam.pose()
    view = np.stack([C[0] - X, C[1] - Y, np.full_like(X, C[2] - sleeve.z_mm)], axis=-1)
    view /= np.linalg.norm(view, axis=-1, keepdims=True)

    d = np.asarray(sleeve.glare_light, float)
    d /= np.linalg.norm(d)
    refl = d - 2 * np.sum(n * d, axis=-1, keepdims=True) * n
    angle = np.arccos(np.clip(np.sum(refl * view, axis=-1), -1, 1))
    glare = np.exp(-0.5 * (angle / np.radians(sleeve.roughness_deg)) ** 2).astype(np.float32)

    cos_view = np.clip(np.sum(n * view, axis=-1), 0, 1)
    sheen = (0.04 + 0.96 * (1 - cos_view) ** 5) * 120  # Schlick Fresnel, reflecting a dim room

    marks = np.zeros((th, tw), np.uint8)
    main_dir = rng.uniform(0, np.pi)  # scratches from handling tend to share a direction
    for _ in range(sleeve.scratches):
        p = np.array([rng.uniform(0, tw), rng.uniform(0, th)])
        a = main_dir + rng.normal(0, 0.3) if rng.uniform() < 0.6 else rng.uniform(0, np.pi)
        length = rng.uniform(0.5, 12) * PX_PER_MM
        q = p + length * np.array([np.cos(a), np.sin(a)])
        cv2.line(marks, tuple(int(c) for c in p), tuple(int(c) for c in q), int(rng.integers(70, 230)),
                 int(rng.choice([1, 1, 1, 2])), cv2.LINE_AA)
    for _ in range(sleeve.scuffs):
        centre = (int(rng.uniform(0, tw)), int(rng.uniform(0, th)))
        for _ in range(int(rng.integers(5, 15))):
            r = int(rng.uniform(1.5, 8) * PX_PER_MM)
            start = rng.uniform(0, 360)
            cv2.ellipse(marks, centre, (r, int(r * rng.uniform(0.6, 1.0))), 0, start, start + rng.uniform(20, 90),
                        int(rng.integers(30, 110)), 1, cv2.LINE_AA)
    for _ in range(int(rng.integers(0, 15))):
        cv2.circle(marks, (int(rng.uniform(0, tw)), int(rng.uniform(0, th))), int(rng.integers(1, 3)),
                   int(rng.integers(80, 255)), -1, cv2.LINE_AA)
    marks = cv2.GaussianBlur(marks.astype(np.float32) / 255, (0, 0), 0.6)

    patches = np.clip(0.5 + 0.9 * smooth_noise((th, tw), rng, cell=max(8, max(th, tw) // 4)), 0, 1)
    fog = (sleeve.fog * patches).astype(np.float32)

    edge = np.zeros((th, tw), np.uint8)
    cv2.rectangle(edge, (0, 0), (tw - 1, th - 1), 255, 2)
    edge = cv2.GaussianBlur(edge.astype(np.float32) / 255, (0, 0), 1.0)

    added = (sleeve.haze * 180 + fog * 45 + sheen + sleeve.gloss * 255 * glare
             + marks * (45 + 150 * glare) + edge * (35 + 120 * glare)).astype(np.float32)
    trans = (1 - sleeve.haze) * ((1 - 0.2 * fog) * (1 - 0.25 * edge))[..., None] * np.asarray(sleeve.tint_bgr, np.float32)
    return A, trans.astype(np.float32), np.repeat(added[..., None], 3, axis=-1), glare, fog


def composite_sleeve(photo, sleeve, cam, label_mm):
    """Lay the film over the photo (uint8, in place). Returns the glare map in image space (0-1).

    Fogged patches scatter light sideways, so what's under them is blurred (by about
    fog_blur_mm at the paper) and whitened."""
    A, trans, added, glare, fog = render_sleeve(sleeve, cam, label_mm)
    Hs = plane_homography(cam, sleeve.z_mm, A)
    size = (photo.shape[1], photo.shape[0])
    warp = lambda img, fill: cv2.warpPerspective(img, Hs, size, flags=cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    under = photo.astype(np.float32)
    if sleeve.fog > 0:
        sigma = max(0.5, sleeve.fog_blur_mm * cam.K[0, 0] / cam.distance_mm)
        f = warp(fog, 0)[..., None]
        under = under + f * (cv2.GaussianBlur(under, (0, 0), sigma) - under)
    out = under * warp(trans, (1, 1, 1)) + warp(added, (0, 0, 0))
    photo[:] = np.clip(out, 0, 255).astype(np.uint8)
    return warp(glare * sleeve.gloss, 0)
