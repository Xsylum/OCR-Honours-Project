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

    def at(self, z):
        """Point on the (straight, slightly leaning) shaft at height z."""
        lean = np.array([*self.lean_mm, 0.0]) * (-z / self.length_mm)
        return np.array([*self.pierce_mm, 0.0]) + lean + np.array([0.0, 0.0, z])

    def points(self):
        return self.at(0.0), self.at(-self.length_mm), self.at(self.below_mm)

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


def draw_pin_piece(photo, cam, pin, z_from, z_to, occ=None):
    """Draw the shaft between heights z_from and z_to into a uint8 BGR photo (and occ mask)."""
    if abs(z_to - z_from) < 1e-6:
        return
    a, b = pin.at(z_from), pin.at(z_to)
    shaft = rod_polygon(cam, a, b, pin.diameter_mm)
    if shaft is None:
        return
    _fill(photo, shaft, (pin.shaft_gray,) * 3)
    if occ is not None:
        _fill(occ, shaft, 255)
    highlight = rod_polygon(cam, a, b, pin.diameter_mm, width_scale=0.3, offset=0.35)
    if highlight is not None:
        _fill(photo, highlight, (min(255, pin.shaft_gray + 80),) * 3)


def draw_pin_head(photo, cam, pin, occ=None):
    (c,), (z,) = project_points(cam, [pin.at(-pin.length_mm)])
    if z <= NEAR_MM:
        return
    r = cam.K[0, 0] * pin.head_diameter_mm / 2 / z
    _disc(photo, c, r, (pin.head_gray,) * 3)
    if occ is not None:
        _disc(occ, c, r, 255)
    _disc(photo, c - np.array([0.35, 0.35]) * r, 0.25 * r, (min(255, pin.head_gray + 120),) * 3)


def label_shadow(cam, A, z, tex_size, light_dir, size):
    """Soft shadow (float 0-1) a stacked label at height z < 0 casts on the target plane z = 0."""
    tw, th = tex_size
    corners = (A @ np.array([[0, 0, 1], [tw, 0, 1], [tw, th, 1], [0, th, 1]], float).T).T
    light = np.asarray(light_dir, float)
    P = np.column_stack([corners[:, :2], np.full(4, z)]) + (-z / light[2]) * light
    pts, depth = project_points(cam, P)
    mask = np.zeros(size[::-1], np.uint8)
    if np.any(depth <= NEAR_MM):
        return mask.astype(np.float32)
    _fill(mask, pts, 255)
    sigma = float(np.clip(cam.K[0, 0] * (0.15 + 0.08 * abs(z)) / depth.mean(), 2.0, 20.0))
    return cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), sigma)


def plane_homography(cam, z, A):
    """Homography for a plane parallel to the label at height z.

    A (3x3) maps the plane's texture pixels to world (X, Y, 1) on that plane.
    """
    R, t, _ = cam.pose()
    H = cam.K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * z + t]) @ A
    return H / H[2, 2]


@dataclass
class StackedLabel:
    """Another label on the same pin, parallel to the target label at height z_mm.

    Negative z is above the target (between it and the camera, so it can hide text);
    positive z is below (only visible around the target's edges).
    """
    z_mm: float
    width_mm: float
    angle_deg: float  # rotation about the pin relative to the target label
    pin_uv: tuple  # where the pin goes through this label, as fractions of its width/height
    source: str  # "printed" or the image_file of another label used as its face

    def world_transform(self, pin, tex_w, tex_h):
        """A: texture pixel -> world (X, Y, 1) on this label's plane."""
        m = self.width_mm / tex_w
        a = np.radians(self.angle_deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        centre = pin.at(self.z_mm)[:2]
        pu, pv = self.pin_uv
        A = np.eye(3)
        A[:2, :2] = rot * m
        A[:2, 2] = centre - rot @ (m * np.array([pu * tex_w, pv * tex_h]))
        return A

    def to_json(self):
        return {"type": "stacked_label", **{k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}}


def sample_stack(rng, pin, label_mm, sources, max_above_mm, max_labels=3):
    """1..max_labels other labels on the pin, above and/or below the target; extends the pin to fit."""
    n = int(rng.integers(1, max_labels + 1))
    n_above = int(rng.integers(0, n + 1))
    stack, z_up, z_down = [], 0.0, 0.0
    for i in range(n):
        above = i < n_above
        gap = float(rng.uniform(2.5, 6.0))
        if above:
            if z_up - gap < -max_above_mm:
                continue
            z_up -= gap
            z = z_up
        else:
            z_down += gap
            z = z_down
        source = "printed" if (not sources or rng.uniform() < 0.5) else str(rng.choice(sources))
        stack.append(StackedLabel(z_mm=float(z), width_mm=float(label_mm[0] * rng.uniform(0.75, 1.2)),
                                  angle_deg=float(rng.normal(0, 18)),
                                  pin_uv=(float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.3, 0.7))),
                                  source=source))
    # The pin has to pass through every label: head above the top one, tip below the bottom one.
    if stack:
        top = min(s.z_mm for s in stack)
        bottom = max(s.z_mm for s in stack)
        pin.length_mm = max(pin.length_mm, -top + float(rng.uniform(2, 6)))
        pin.below_mm = max(pin.below_mm, bottom + float(rng.uniform(1, 3)))
    return stack


@dataclass
class BeetleBody:
    """Ellipsoid standing in for the pinned insect above the label.

    Floating a millimetre or two above the paper means an oblique camera can see text that a
    top-down view has hidden: the same 3D parallax the pin already uses. `radii_mm` are the
    semi-axes along the body's local length, width and height.
    """
    centre_mm: tuple  # (x, y, z); z < 0 = above the paper
    radii_mm: tuple  # (length/2, width/2, height/2) in mm
    yaw_deg: float = 0.0  # rotation about the label normal
    pitch_deg: float = 0.0  # slight nose-up / nose-down
    color_bgr: tuple = (35, 50, 70)

    @property
    def rotation(self):
        y, p = np.radians(self.yaw_deg), np.radians(self.pitch_deg)
        cy, sy, cp, sp = np.cos(y), np.sin(y), np.cos(p), np.sin(p)
        # Local axes: +x along the body, +y across, +z toward the paper (world +z).
        return np.array([[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0.0, cp]])

    def bottom_z(self):
        """Highest z of the body (closest to the paper); still negative when floating."""
        return float(self.centre_mm[2] + self.radii_mm[2] * abs(self.rotation[2, 2]))

    def top_z(self):
        return float(self.centre_mm[2] - self.radii_mm[2] * abs(self.rotation[2, 2]))

    def surface_points(self, n_u=28, n_v=16):
        u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
        v = np.linspace(0, np.pi, n_v)
        uu, vv = np.meshgrid(u, v)
        a, b, c = self.radii_mm
        local = np.stack([a * np.sin(vv) * np.cos(uu), b * np.sin(vv) * np.sin(uu), c * np.cos(vv)], -1)
        return local.reshape(-1, 3) @ self.rotation.T + np.asarray(self.centre_mm, float)

    def silhouette(self, cam):
        """Convex image polygon of the projected body (None if entirely behind the near plane)."""
        pts, depth = project_points(cam, self.surface_points())
        keep = depth > NEAR_MM
        if keep.sum() < 3:
            return None
        hull = cv2.convexHull(pts[keep].astype(np.float32))
        return hull.reshape(-1, 2)

    def to_json(self):
        return {"type": "beetle", **{k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}}


def sample_beetle(rng, label_mm, pin=None, cover=(0.12, 0.95)):
    """Random beetle above the label whose top-down footprint covers `cover` of the label area.

    Sizes range from a small oval on one part of the name up to nearly the whole label. The
    centre is placed somewhere on the label (small beetles wander more; large ones stay
    nearer the middle). Anchored near the pin when one is present. Extends the pin so its
    head sits above the body.
    """
    W, H = label_mm
    # Mix of small / medium / label-sized so the dataset isn't one repeated oval.
    pick = float(rng.uniform())
    if cover == (0.12, 0.95):
        if pick < 0.35:
            frac = float(rng.uniform(0.12, 0.28))  # small
        elif pick < 0.70:
            frac = float(rng.uniform(0.30, 0.55))  # medium
        else:
            frac = float(rng.uniform(0.70, 0.95))  # nearly as big as the label
    else:
        frac = float(rng.uniform(*cover))
    # Scale the inscribed ellipse (π/4 of the rectangle); stretch axes a bit for shape variety.
    scale = float(np.sqrt(frac / (np.pi / 4)))
    a = 0.5 * W * scale * float(rng.uniform(0.85, 1.15))
    b = 0.5 * H * scale * float(rng.uniform(0.85, 1.15))
    c = float(rng.uniform(0.6, 1.4))
    # Above the paper so an oblique camera can look under the body and read the label.
    gap = float(rng.uniform(2.0, 3.5))
    # How far the centre can wander while keeping most of the oval over the label.
    max_ox = max(0.02 * W, 0.5 * W - 0.35 * a)
    max_oy = max(0.02 * H, 0.5 * H - 0.35 * b)
    if pin is not None:
        cx = float(np.clip(pin.pierce_mm[0] + rng.uniform(-max_ox, max_ox) * 0.5, -max_ox, max_ox))
        cy = float(np.clip(pin.pierce_mm[1] + rng.uniform(-max_oy, max_oy) * 0.5, -max_oy, max_oy))
    else:
        cx = float(rng.uniform(-max_ox, max_ox))
        cy = float(rng.uniform(-max_oy, max_oy))
    centre_z = -(c + gap)
    # Flat on the label plane (no pitch). About half the time the long axis runs along the
    # name; the other half it's turned 90° so the oval sits across the label instead.
    yaw = 0.0 if rng.uniform() < 0.5 else float(rng.choice([90.0, -90.0]))
    beetle = BeetleBody(centre_mm=(float(cx), float(cy), float(centre_z)),
                        radii_mm=(float(a), float(b), float(c)),
                        yaw_deg=yaw,
                        pitch_deg=0.0,
                        color_bgr=tuple(int(v) for v in rng.integers([20, 30, 40], [55, 80, 110])))
    if pin is not None:
        pin.length_mm = max(pin.length_mm, -beetle.top_z() + float(rng.uniform(1.5, 4.0)))
    return beetle


def beetle_shadow(cam, beetle, light_dir, size):
    """Soft shadow (float 0-1) the beetle casts on the label plane z = 0."""
    light = np.asarray(light_dir, float)
    pts = beetle.surface_points()
    # Walk each surface point along the light ray down onto the paper.
    s = -pts[:, 2:3] / light[2]
    on_paper = pts + s * light
    uv, depth = project_points(cam, on_paper)
    mask = np.zeros(size[::-1], np.uint8)
    keep = depth > NEAR_MM
    if keep.sum() < 3:
        return mask.astype(np.float32)
    hull = cv2.convexHull(uv[keep].astype(np.float32)).reshape(-1, 2)
    _fill(mask, hull, 255)
    mean_z = float(np.mean(-pts[:, 2]))
    sigma = float(np.clip(cam.K[0, 0] * (0.2 + 0.05 * mean_z) / max(depth[keep].mean(), NEAR_MM), 2.0, 18.0))
    return cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), sigma)


def draw_beetle(photo, cam, beetle, light_dir=None, occ=None):
    """Draw the beetle silhouette into a uint8 BGR photo (and occlusion mask)."""
    poly = beetle.silhouette(cam)
    if poly is None:
        return
    base = np.array(beetle.color_bgr, np.int32)
    # Shade in a local patch so the highlight stays clipped to the body.
    pad = 3
    x0, y0 = np.clip(np.floor(poly.min(axis=0) - pad).astype(int), 0, None)
    x1, y1 = np.clip(np.ceil(poly.max(axis=0) + pad).astype(int), None, [photo.shape[1], photo.shape[0]])
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    _fill(m, poly - [x0, y0], 255)
    body = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    body[:] = base
    centre = poly.mean(axis=0) - [x0, y0]
    extent = float(np.linalg.norm(poly - poly.mean(axis=0), axis=1).max())
    if light_dir is not None:
        L = cam.pose()[0] @ np.asarray(light_dir, float)
        shift = np.array([-L[0], -L[1]])
        n = np.linalg.norm(shift)
        shift = shift / n if n > 1e-6 else np.array([-1.0, -1.0])
    else:
        shift = np.array([-1.0, -1.0])
    # Radial falloff from a highlight shifted toward the light → round shaded body.
    yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
    hi = centre + 0.3 * extent * shift
    r = np.sqrt((xx - hi[0]) ** 2 + (yy - hi[1]) ** 2) / max(extent, 1e-3)
    shade = np.clip(1.25 - 0.7 * r, 0.55, 1.35)
    body = np.clip(body.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)
    mask = m.astype(bool)
    photo[y0:y1, x0:x1][mask] = body[mask]
    if occ is not None:
        occ[y0:y1, x0:x1][mask] = 255
    # Soften the hard rim so it reads as a round body rather than a cut-out.
    soft = cv2.GaussianBlur(photo[y0:y1, x0:x1], (0, 0), 1.2)
    edge = cv2.dilate(m, np.ones((3, 3), np.uint8)) - m
    photo[y0:y1, x0:x1] = np.where(edge[..., None] > 0, soft, photo[y0:y1, x0:x1])


LOCALITY_WORDS = ["CANADA", "ON", "QC", "Ottawa", "Gatineau", "Algonquin Pk", "USA", "NY", "Adirondack Mts",
                  "Mer Bleue", "pitfall trap", "UV light", "sweeping", "leg.", "coll.", "Malaise trap",
                  "mixed forest", "bog", "under bark", "Berlese", "DET.", "CNC"]
MONTHS = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]


def printed_label(rng, width_px=420):
    """A typed collection label (locality / date / collector) to use as a stacked label's face."""
    lines = int(rng.integers(2, 5))
    height = int(width_px * rng.uniform(0.28, 0.42))
    img = np.full((height, width_px, 3), int(rng.uniform(225, 250)), np.uint8)
    scale = 0.5 * width_px / 420
    y = int(height / (lines + 1))
    for i in range(lines):
        if i == 1:
            text = f"{int(rng.integers(1, 29))}.{rng.choice(MONTHS)}.{int(rng.integers(1950, 2025))}"
        else:
            text = " ".join(rng.choice(LOCALITY_WORDS, size=int(rng.integers(2, 4)), replace=False))
        cv2.putText(img, text, (int(width_px * 0.05), y * (i + 1) + 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (25, 25, 25), 1, cv2.LINE_AA)
    return img
