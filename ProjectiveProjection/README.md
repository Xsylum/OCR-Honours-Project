# Projective Projection

Simulates photographing a finished label with a real camera from different distances and oblique angles, and moves the OCR boxes along with the pixels.

```
python -m ProjectiveProjection.camera --views 3
```

Input (from `label_generation_pipeline.py`): `master_dataset_labels.json` and the images in `final_output_labels/`.
Output: `projected_output_labels/label_<n>_v<k>.png`, one `projected_dataset_labels.json` with the warped boxes, and `preview_sheet.png` with the boxes drawn on the first views.

Options: `--labels`, `--images`, `--out`, `--views`, `--seed`, `--label-width-mm` (default 22), `--image-size` (default 1024 768), `--pin-prob` (default 0.8), `--no-effects`, `--save-masks`.

For clean geometry only (no pin, no effects): `--no-effects --pin-prob 0`.

Example output is in [examples/](examples/); start with `examples/stages.jpg`.

## How it works

The label is treated as a flat sheet of paper lying at `z = 0`, measured in millimetres (the label image is scaled so it is `--label-width-mm` wide). A virtual pinhole camera is placed above it:

- **Intrinsics `K`**: a 50 mm lens on a 36 mm-wide sensor, converted to a focal length in pixels; the principal point is the image centre.
- **Extrinsics `R`, `t`**: the camera sits at distance `d` from a point on the label, tilted `tilt` degrees away from straight-down in direction `azimuth`, looking at that point, then rolled about its own axis by `roll`.
- Because the label is flat, the whole projection reduces to one 3x3 **homography** `H = K [r1 r2 t] S`, where `S` converts label pixels to millimetres. OpenCV's `warpPerspective` applies `H` to the image and `perspectiveTransform` applies it to the box corners.

Each view samples a random camera: the label fills 40-85% of the frame width, tilt 0-50 degrees (biased towards oblique), any azimuth, roll within 10 degrees, a small off-centre aim point and a random background gray. Seeds are derived from the image name, so runs are reproducible.

Output boxes are 4-point quads (a rectangle stops being a rectangle under perspective), clockwise from top-left. Each entry stores the camera (`K`, `R`, `t`, distance, angles), `H`, the occluders and the effect parameters, so a view can be reproduced exactly.

## Pin occluder (`occluders.py`)

Each view gets an insect pin through the label with probability `--pin-prob`. The pin is real 3D geometry in the same world as the label:

- A steel shaft from a few mm below the paper up to a ball head 5-15 mm above it, with a random slight lean, diameter and head color.
- Its two ends are projected with the same camera, and the shaft's width follows perspective. As a result the pin covers different text from different angles, which a flat 2D overlay can't do.
- It casts a soft shadow on the label from the view's light direction (projected onto the paper plane).
- The shaft below the paper is drawn only where the label isn't covering it.
- Geometry closer than 3 mm to the camera is clipped.

The pin's pixels form an occlusion mask (`--save-masks` writes it out). Each box gets:

- `visible_fraction`: the share of the box that is in the photo and not covered.
- `occluded_fraction`: the share of the in-frame box covered by the pin.

## Realism effects (`effects.py`)

Applied in physical order, with every parameter stored under `effects`:

1. Paper: uneven tone, slight yellowing and grain on the label (before warping, so it moves with the paper).
2. Background: a textured surface (foam, cork or drawer-like) instead of flat gray.
3. Lighting: a brightness gradient aligned with the light direction, vignetting and an exposure change.
4. White balance shift.
5. Blur: defocus (Gaussian), plus motion blur in about 25% of views.
6. Sensor noise: read noise plus brightness-dependent shot noise.
7. JPEG compression (quality 50-95).

Tests: `python -m pytest tests`.

## Not done yet

- Other occluders: the beetle itself above the label, and stacked labels on the same pin. They would use the same approach as the pin, e.g. an ellipsoid beetle body projected through the camera.
- Lens distortion and paper curl (the label is perfectly flat here).
