# Projective Projection

Simulates photographing a finished label with a real camera from different distances and oblique angles, and moves the OCR boxes along with the pixels.

```
python -m ProjectiveProjection.camera --views 3
```

Input (from `label_generation_pipeline.py`): `master_dataset_labels.json` and the images in `final_output_labels/`.
Output: `projected_output_labels/label_<n>_v<k>.png`, one `projected_dataset_labels.json` with the warped boxes, and `preview_sheet.png` with the boxes drawn on the first views.

Options: `--labels`, `--images`, `--out`, `--views`, `--seed`, `--label-width-mm` (default 22), `--image-size` (default 1024 768), `--pin-prob` (default 0.8), `--stack-prob` (default 0, off), `--max-stack` (default 3), `--sleeve-prob` (default 0.4), `--distortion` (default 1), `--no-effects`, `--save-masks`.

For clean geometry only (no pin, sleeve, distortion or effects): `--no-effects --pin-prob 0`.

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

## Stacked labels

**Off by default for now** (`--stack-prob 0`); pass e.g. `--stack-prob 0.5` to turn it on.

Real specimens have several labels on one pin (locality, date, determination...). With probability `--stack-prob`, a pinned view gets 1 to `--max-stack` (default 3) other labels on the same pin:

- Each one is a plane parallel to the target, 2.5-6 mm above or below the previous one, rotated about the pin and pierced at a random point, with its own homography `K [r1 r2 (r3 z + t)] A`.
- Its face is either a generated typed locality/date label or another label from the same dataset (a realistic distractor for OCR).
- Everything is drawn back to front (painter's algorithm, exact for parallel planes), with the pin drawn in pieces between the labels. Labels below the target only show around its edges. Labels above it can hide text and cast soft shadows on it.
- The pin is lengthened so it passes through every label. Labels too close to the camera are skipped.

Stacked labels appear in `occluders` with `type: "stacked_label"`.

## Plastic sleeve (`sleeve.py`)

With probability `--sleeve-prob` (default 0.4), the label is inside a clear protective film: a plane 0.2-0.8 mm above the paper, overhanging it by 0.5-2.5 mm and slightly crooked. It is drawn with the same camera, as a layer in the back-to-front order (so the pin and labels above it are drawn over it):

- **Haze and tint**: the film scatters 2-8% of the light, lifting the blacks and lowering text contrast, with a faint color cast.
- **Glare**: a light source mirrors off the film. For every point of the film, the reflected light direction is compared with the direction to the camera, using the normal of a height field (gentle waviness plus a few creases). The highlight therefore moves with the viewpoint and breaks into streaks along creases. The light is aimed so the highlight lands somewhere around the label, often partly off it.
- **Fog and smudges**: patchy cloudy areas scatter light sideways, so the text under them is blurred (by 0.04-0.15 mm at the paper) and slightly whitened.
- **Scratches and scuffs**: 5-60 fine hairline scratches (mostly sharing one direction, like handling marks) and a few swirl marks from wiping. They catch light, so they get brighter inside the glare.
- **Fresnel sheen** that grows at grazing angles, a bright cut edge and dust.

The film isn't an occluder (text stays behind it), but glare can wash text out, so each box also gets `glare_fraction`: the share of the box where the glare is strong. The sleeve's parameters are stored under `sleeve`.

## Lens distortion (`lens.py`)

Views with effects get radial and tangential lens distortion (OpenCV's Brown-Conrady model: `k1`, `k2`, `p1`, `p2`, `k3`). It is mostly barrel distortion with `k1` between -0.3 and 0.12; `--distortion` scales it (0 = ideal pinhole). The finished scene (photo, occlusion mask, glare map) is resampled through the model, and every output quad goes through the same model, so boxes stay on the ink. The coefficients are stored under `camera.distortion` in OpenCV order: `cv2.undistort(photo, K, opencv_dist_coeffs)` recovers the pinhole view, where `homography` holds exactly. Distorted quads use the distorted corners; their edges are really slightly curved, which is negligible at word-box size.

## Occlusion

The pin's pixels and the labels above the target form an occlusion mask (`--save-masks` writes it out). Each box gets:

- `visible_fraction`: the share of the box that is in the photo and not covered.
- `occluded_fraction`: the share of the in-frame box covered by the pin or a label above it.
- `glare_fraction`: the share of the in-frame box washed out by sleeve glare (0 without a sleeve).

In `preview_sheet.png`, red boxes are occluded, magenta boxes have glare and blue boxes are clear.

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

- The beetle itself above the label. It would use the same approach as the pin, e.g. an ellipsoid body projected through the camera.
- Paper curl (the label and sleeve are flat planes here).
