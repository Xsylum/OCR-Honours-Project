"""Build the example outputs in this folder (for demos / meetings).

    python -m ProjectiveProjection.examples.make_examples

- long_names/: labels in long_names/inputs/ (images only, no OCR boxes), so word boxes are
  found from the ink. Each label gets several camera views.
- sample_labels/: the label generator's committed sample (master_dataset_labels.json +
  final_output_labels/), one view per label.
- stages.png: one label through the stages: original -> clean projection -> pin + effects.

Example photos are saved as JPEG to keep the repo small.
"""

import json
import random
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ProjectiveProjection.camera import (box_to_quad, in_front, label_to_world, project_dataset, render_view,
                                         sample_camera)
from ProjectiveProjection.occluders import sample_pin

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def word_boxes(image, words):
    """[x, y, w, h] per word: split the ink at the len(words) - 1 widest column gaps."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    cols = np.flatnonzero(ink.any(axis=0))
    gaps = [(b - a, a, b) for a, b in zip(cols, cols[1:]) if b - a > 1]
    cuts = sorted(sorted(gaps, reverse=True)[:len(words) - 1], key=lambda g: g[1])
    spans, start = [], cols[0]
    for _, a, b in cuts:
        spans.append((start, a + 1))
        start = b
    spans.append((start, cols[-1] + 1))
    boxes = []
    for text, (x0, x1) in zip(words, spans):
        rows = np.flatnonzero(ink[:, x0:x1].any(axis=1))
        pad = 3
        y0, y1 = max(rows[0] - pad, 0), min(rows[-1] + 1 + pad, image.shape[0])
        x0, x1 = max(x0 - pad, 0), min(x1 + pad, image.shape[1])
        boxes.append({"text": text, "box": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]})
    return boxes


def to_jpeg(out_dir, quality=90):
    """Convert the projected PNGs to JPEG and point the JSON at them."""
    meta_path = out_dir / "projected_dataset_labels.json"
    meta = json.loads(meta_path.read_text())
    for view in meta:
        png = out_dir / view["image_file"]
        jpg = png.with_suffix(".jpg")
        cv2.imwrite(str(jpg), cv2.imread(str(png)), [cv2.IMWRITE_JPEG_QUALITY, quality])
        png.unlink()
        view["image_file"] = jpg.name
    meta_path.write_text(json.dumps(meta, indent=2))
    sheet = out_dir / "preview_sheet.png"
    if sheet.exists():
        cv2.imwrite(str(sheet.with_suffix(".jpg")), cv2.imread(str(sheet)), [cv2.IMWRITE_JPEG_QUALITY, quality])
        sheet.unlink()


def long_names(views=3):
    inputs = HERE / "long_names" / "inputs"
    entries = []
    for path in sorted(inputs.glob("*.png")):
        name = " ".join(path.stem.split("_")[2:])  # label_<n>_<Genus>_<species>.png
        image = cv2.imread(str(path))
        entries.append({"image_file": path.name, "ground_truth": name, "words": word_boxes(image, name.split())})
    labels_json = HERE / "long_names" / "labels.json"
    labels_json.write_text(json.dumps(entries, indent=2))
    out = HERE / "long_names" / "projected"
    shutil.rmtree(out, ignore_errors=True)
    project_dataset(labels_json, inputs, out, views=views, seed=1)
    to_jpeg(out)
    return entries


def sample_labels():
    out = HERE / "sample_labels"
    shutil.rmtree(out, ignore_errors=True)
    project_dataset(ROOT / "master_dataset_labels.json", ROOT / "final_output_labels", out, views=1, seed=0)
    to_jpeg(out)


def stages(entry, seed=4, height=300):
    """Original label | clean projection | same camera with pin + realism effects (boxes drawn)."""
    image = cv2.imread(str(HERE / "long_names" / "inputs" / entry["image_file"]))
    h, w = image.shape[:2]
    regions = [{"text": wd["text"], "quad": box_to_quad(wd["box"])} for wd in entry["words"]]
    label_mm = (22.0, 22.0 * h / w)
    rng = random.Random(seed)
    for _ in range(50):
        cam = sample_camera(rng, label_mm, max_tilt=45)
        if cam.tilt_deg > 25 and in_front(cam, label_to_world(w, h, 22.0 / w), w, h):
            break
    nrng = np.random.default_rng(seed)
    pin = sample_pin(nrng, label_mm)
    clean, *_ = render_view(image, regions, cam, np.random.default_rng(seed), effects=False)
    real, warped, _, _ = render_view(image, regions, cam, nrng, effects=True, pin=pin)
    boxed = real.copy()
    for r in warped:
        color = (0, 0, 255) if r["occluded_fraction"] > 0.02 else (255, 0, 0)
        cv2.polylines(boxed, [np.int32(r["quad"])], True, color, 2, cv2.LINE_AA)

    original = image.copy()
    for wd in entry["words"]:
        x, y, bw, bh = wd["box"]
        cv2.rectangle(original, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
    panels = []
    for title, img in [("1. generated label + word boxes", original), ("2. virtual camera (clean)", clean),
                       ("3. + pin, shadow, lighting, blur, noise", real), ("4. ground-truth boxes after projection", boxed)]:
        img = cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height), interpolation=cv2.INTER_AREA)
        canvas = np.full((height + 34, img.shape[1], 3), 255, np.uint8)
        canvas[34:] = img
        cv2.putText(canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
        panels.append(canvas)
    gap = np.full((height + 34, 12, 3), 255, np.uint8)
    row = panels[0]
    for p in panels[1:]:
        row = np.hstack([row, gap, p])
    cv2.imwrite(str(HERE / "stages.jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    entries = long_names()
    sample_labels()
    stages(entries[0])
    print(f"examples written to {HERE}")


if __name__ == "__main__":
    main()
