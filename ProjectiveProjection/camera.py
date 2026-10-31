"""Projective projection: photograph a flat label with a virtual pinhole camera.

The label is a plane at z = 0 (world units: mm, x right, y down, camera above the paper on
the -z side). A camera at distance d, tilted away from the label normal, looks at a point on
the label. A label pixel maps to the photo by the homography H = K [r1 r2 t] S.

Reads the label generator's output (master_dataset_labels.json + final_output_labels/) and
writes several camera views per label plus one JSON with the warped OCR boxes:

    python -m ProjectiveProjection.camera --views 3
"""

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Camera:
    distance_mm: float = 60.0
    tilt_deg: float = 0.0  # angle between optical axis and label normal
    azimuth_deg: float = 0.0  # direction of the tilt, in the label plane
    roll_deg: float = 0.0  # rotation about the optical axis
    focal_mm: float = 50.0
    sensor_width_mm: float = 36.0
    image_size: tuple = (1024, 768)
    target_mm: tuple = (0.0, 0.0)  # point on the label the camera looks at (label centre = 0, 0)

    @property
    def K(self):
        w, h = self.image_size
        f = self.focal_mm * w / self.sensor_width_mm
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)

    def pose(self):
        """World -> camera rotation R, translation t and camera centre C."""
        th, ph = np.radians(self.tilt_deg), np.radians(self.azimuth_deg)
        target = np.array([*self.target_mm, 0.0])
        C = target + self.distance_mm * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), -np.cos(th)])
        z = (target - C) / np.linalg.norm(target - C)
        x = np.cross([0.0, 1.0, 0.0], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        r = np.radians(self.roll_deg)
        roll = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
        R = roll @ np.vstack([x, y, z])
        return R, -R @ C, C


def label_to_world(width_px, height_px, mm_per_px):
    """S: label pixel (u, v, 1) -> world plane point (X, Y, 1), label centred at the origin."""
    m = mm_per_px
    return np.array([[m, 0, -m * width_px / 2], [0, m, -m * height_px / 2], [0, 0, 1]], np.float64)


def homography(cam, S):
    R, t, _ = cam.pose()
    H = cam.K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
    return H / H[2, 2]


def in_front(cam, S, width_px, height_px):
    R, t, _ = cam.pose()
    corners = np.array([[0, 0, 1], [width_px, 0, 1], [width_px, height_px, 1], [0, height_px, 1]], float)
    world = (S @ corners.T).T
    world[:, 2] = 0
    return bool(np.all((R @ world.T).T[:, 2] + t[2] > 1.0))


def sample_camera(rng, label_mm, image_size=(1024, 768), focal_mm=50.0, sensor_width_mm=36.0,
                  fill=(0.4, 0.85), max_tilt=50.0, max_roll=10.0):
    """Random but plausible 'photo of a label' camera: label spans `fill` of the frame width."""
    f_px = focal_mm * image_size[0] / sensor_width_mm
    d = f_px * label_mm[0] / (rng.uniform(*fill) * image_size[0])
    return Camera(distance_mm=d,
                  tilt_deg=max_tilt * np.sqrt(rng.uniform(0, 1)),  # more mass at oblique angles
                  azimuth_deg=rng.uniform(0, 360),
                  roll_deg=rng.uniform(-max_roll, max_roll),
                  focal_mm=focal_mm, sensor_width_mm=sensor_width_mm, image_size=tuple(image_size),
                  target_mm=(rng.uniform(-0.1, 0.1) * label_mm[0], rng.uniform(-0.1, 0.1) * label_mm[1]))


def visible_fraction(quad, image_size):
    q = np.asarray(quad, np.float32)
    area = abs(cv2.contourArea(q))
    if area < 1e-6:
        return 0.0
    w, h = image_size
    frame = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    inter, _ = cv2.intersectConvexConvex(q, frame)
    return float(min(1.0, inter / area))


def box_to_quad(box):
    """[x, y, w, h] -> 4 corners, clockwise from top-left."""
    x, y, w, h = box
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def warp_quad(H, quad):
    pts = np.asarray(quad, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _ql(q):
    return [[round(float(x), 2), round(float(y), 2)] for x, y in q]


def project_label(image, regions, cam, label_width_mm=22.0, background=110):
    """Warp a label image and its text regions ({"text", "quad"}) into one camera view.

    Returns (photo, warped regions with visible_fraction, info dict with H and camera).
    """
    h, w = image.shape[:2]
    S = label_to_world(w, h, label_width_mm / w)
    H = homography(cam, S)
    border = (int(background),) * (image.shape[2] if image.ndim == 3 else 1)
    photo = cv2.warpPerspective(image, H, tuple(cam.image_size), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    size = list(cam.image_size)
    warped = []
    for r in regions:
        q = warp_quad(H, r["quad"])
        warped.append({"text": r["text"], "quad": _ql(q), "visible_fraction": round(visible_fraction(q, size), 3)})
    R, t, C = cam.pose()
    info = {
        "image_size": size,
        "label_quad": _ql(warp_quad(H, [[0, 0], [w, 0], [w, h], [0, h]])),
        "homography": H.round(8).tolist(),
        "camera": {**asdict(cam), "K": cam.K.round(4).tolist(), "R": R.round(8).tolist(),
                   "t": t.round(6).tolist(), "center_mm": C.round(4).tolist(), "label_width_mm": label_width_mm},
    }
    return photo, warped, info


def _seed(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def project_dataset(labels_json, images_dir, out_dir, views=3, seed=0, label_width_mm=22.0,
                    image_size=(1024, 768)):
    """Project every label in master_dataset_labels.json; writes images + projected_dataset_labels.json."""
    entries = json.loads(Path(labels_json).read_text())
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset, previews = [], []
    for entry in entries:
        image = cv2.imread(str(images_dir / entry["image_file"]), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  skip {entry['image_file']}: image not found in {images_dir}")
            continue
        h, w = image.shape[:2]
        regions = [{"text": wd["text"], "quad": box_to_quad(wd["box"])} for wd in entry["words"]]
        label_mm = (label_width_mm, label_width_mm * h / w)
        stem = Path(entry["image_file"]).stem
        for k in range(views):
            vseed = _seed(seed, entry["image_file"], k)
            rng = random.Random(vseed)
            for _ in range(20):
                cam = sample_camera(rng, label_mm, image_size)
                if in_front(cam, label_to_world(w, h, label_width_mm / w), w, h):
                    break
            photo, warped, info = project_label(image, regions, cam, label_width_mm,
                                                background=rng.uniform(60, 180))
            name = f"{stem}_v{k}.png"
            cv2.imwrite(str(out_dir / name), photo)
            dataset.append({"image_file": name, "source_image": entry["image_file"],
                            "ground_truth": entry["ground_truth"], "seed": vseed, "words": warped, **info})
            if len(previews) < 12:
                previews.append(_preview(photo, warped, info))

    (out_dir / "projected_dataset_labels.json").write_text(json.dumps(dataset, indent=2))
    if len(previews) >= 3:
        n = len(previews) // 3 * 3
        cv2.imwrite(str(out_dir / "preview_sheet.png"),
                    np.vstack([np.hstack(previews[i:i + 3]) for i in range(0, n, 3)]))
    print(f"[project] {len(dataset)} views of {len(entries)} labels -> {out_dir}")
    return dataset


def _preview(photo, regions, info, width=420):
    vis = photo.copy() if photo.ndim == 3 else cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, [np.int32(info["label_quad"])], True, (0, 200, 255), 2)
    for r in regions:
        cv2.polylines(vis, [np.int32(r["quad"])], True, (255, 0, 0), 2)
    cam = info["camera"]
    vis = cv2.resize(vis, (width, int(vis.shape[0] * width / vis.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.putText(vis, f"d={cam['distance_mm']:.0f}mm tilt={cam['tilt_deg']:.0f} roll={cam['roll_deg']:.0f}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, default=Path("master_dataset_labels.json"))
    p.add_argument("--images", type=Path, default=Path("final_output_labels"))
    p.add_argument("--out", type=Path, default=Path("projected_output_labels"))
    p.add_argument("--views", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label-width-mm", type=float, default=22.0, help="physical width of the label")
    p.add_argument("--image-size", type=int, nargs=2, default=[1024, 768])
    a = p.parse_args(argv)
    project_dataset(a.labels, a.images, a.out, a.views, a.seed, a.label_width_mm, tuple(a.image_size))


if __name__ == "__main__":
    main()
