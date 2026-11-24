"""Photometric realism effects for simulated label photos.

Every function takes a numpy Generator so a view is reproducible from its seed, and returns
the parameters it used so they can be stored next to the image.
"""

import cv2
import numpy as np


def smooth_noise(shape, rng, cell=64):
    """Low-frequency noise in [-1, 1] (random grid upsampled with cubic interpolation)."""
    h, w = shape
    small = rng.normal(size=(h // cell + 3, w // cell + 3)).astype(np.float32)
    big = cv2.resize(small, (w + 2 * cell, h + 2 * cell), interpolation=cv2.INTER_CUBIC)[cell:cell + h, cell:cell + w]
    return big / max(float(np.abs(big).max()), 1e-6)


def paper_texture(label, rng):
    """Uneven paper tone and fibre grain on the label image, before it is warped."""
    h, w = label.shape[:2]
    tone = 1 - rng.uniform(0.02, 0.07) * (smooth_noise((h, w), rng, cell=max(16, w // 8)) * 0.5 + 0.5)
    tint = np.array([rng.uniform(0.95, 1.0), rng.uniform(0.97, 1.0), 1.0], np.float32)  # BGR: slightly yellowed
    grain = rng.normal(0, rng.uniform(1.0, 3.0), (h, w)).astype(np.float32)
    out = label.astype(np.float32) * tone[..., None] * tint + grain[..., None]
    return np.clip(out, 0, 255)


def background(size, rng):
    """Textured surface behind the label (foam / cork / drawer-like)."""
    w, h = size
    base = rng.uniform(55, 185)
    color = base * np.array([rng.uniform(0.85, 1.0), rng.uniform(0.9, 1.0), rng.uniform(0.9, 1.1)], np.float32)
    blotch = smooth_noise((h, w), rng, cell=int(rng.integers(40, 160))) * rng.uniform(5, 20)
    grain = rng.normal(0, rng.uniform(2, 6), (h, w)).astype(np.float32)
    return np.clip(color + (blotch + grain)[..., None], 0, 255)


def sample_light(rng):
    """Unit direction a directional light travels in (towards +z = down onto the label)."""
    elevation = np.radians(rng.uniform(45, 80))  # angle above the paper
    azimuth = rng.uniform(0, 2 * np.pi)
    d = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    return d / np.linalg.norm(d)


def lighting(img, rng, light_dir=None):
    """Brightness gradient (falling off away from the light) plus lens vignetting."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx, yy = (xx - w / 2) / w, (yy - h / 2) / h
    ang = np.arctan2(light_dir[1], light_dir[0]) if light_dir is not None else rng.uniform(0, 2 * np.pi)
    strength = rng.uniform(0.05, 0.25)
    gradient = 1 - strength * (xx * np.cos(ang) + yy * np.sin(ang))
    vignette = 1 - rng.uniform(0.0, 0.35) * (xx ** 2 + yy ** 2) * 2
    exposure = rng.uniform(0.85, 1.1)
    out = img * (gradient * vignette * exposure)[..., None]
    return out, {"gradient_strength": round(float(strength), 3), "exposure": round(float(exposure), 3)}


def white_balance(img, rng):
    gains = np.array([rng.uniform(0.9, 1.1), 1.0, rng.uniform(0.9, 1.1)], np.float32)
    return img * gains, {"wb_gains_bgr": [round(float(g), 3) for g in gains]}


def blur(img, rng):
    """Defocus (Gaussian) and occasional hand-shake (motion) blur."""
    params = {}
    sigma = float(rng.uniform(0.0, 1.6))
    if sigma > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        params["defocus_sigma"] = round(sigma, 2)
    if rng.uniform() < 0.25:
        length = int(rng.integers(3, 9))
        k = np.zeros((length, length), np.float32)
        k[length // 2, :] = 1
        angle = float(rng.uniform(0, 180))
        rot = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), angle, 1)
        k = cv2.warpAffine(k, rot, (length, length))
        img = cv2.filter2D(img, -1, k / max(k.sum(), 1e-6))
        params["motion_blur"] = {"length_px": length, "angle_deg": round(angle, 1)}
    return img, params


def sensor_noise(img, rng):
    """Signal-dependent (shot) + constant (read) noise."""
    read = rng.uniform(1.0, 5.0)
    shot = rng.uniform(0.0, 0.5)
    sigma = np.sqrt(read ** 2 + shot * np.clip(img, 0, 255))
    return img + rng.normal(size=img.shape).astype(np.float32) * sigma, {"read_noise": round(float(read), 2)}


def jpeg(img, rng):
    q = int(rng.integers(50, 96))
    ok, buf = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(np.float32), {"jpeg_quality": q}


def camera_effects(photo, rng, light_dir=None):
    """Apply the post-capture effects in physical order. Returns (uint8 photo, params)."""
    img = photo.astype(np.float32)
    params = {}
    for step in (lambda x: lighting(x, rng, light_dir), lambda x: white_balance(x, rng),
                 lambda x: blur(x, rng), lambda x: sensor_noise(x, rng), lambda x: jpeg(x, rng)):
        img, p = step(img)
        params.update(p)
    return np.clip(img, 0, 255).astype(np.uint8), params
