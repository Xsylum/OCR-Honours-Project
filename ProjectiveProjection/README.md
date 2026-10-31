# Projective Projection

Simulates photographing a finished label with a real camera from different distances and oblique angles, and moves the OCR boxes along with the pixels.

```
python -m ProjectiveProjection.camera --views 3
```

Input (from `label_generation_pipeline.py`): `master_dataset_labels.json` and the images in `final_output_labels/`.
Output: `projected_output_labels/label_<n>_v<k>.png`, one `projected_dataset_labels.json` with the warped boxes, and `preview_sheet.png` with the boxes drawn on the first views.

Options: `--labels`, `--images`, `--out`, `--views`, `--seed`, `--label-width-mm` (default 22), `--image-size` (default 1024 768).

## How it works

The label is treated as a flat sheet of paper lying at `z = 0`, measured in millimetres (the label image is scaled so it is `--label-width-mm` wide). A virtual pinhole camera is placed above it:

- **Intrinsics `K`**: a 50 mm lens on a 36 mm-wide sensor, converted to a focal length in pixels; the principal point is the image centre.
- **Extrinsics `R`, `t`**: the camera sits at distance `d` from a point on the label, tilted `tilt` degrees away from straight-down in direction `azimuth`, looking at that point, then rolled about its own axis by `roll`.
- Because the label is flat, the whole projection reduces to one 3x3 **homography** `H = K [r1 r2 t] S`, where `S` converts label pixels to millimetres. OpenCV's `warpPerspective` applies `H` to the image and `perspectiveTransform` applies it to the box corners.

Each view samples a random camera: the label fills 40-85% of the frame width, tilt 0-50 degrees (biased towards oblique), any azimuth, roll within 10 degrees, a small off-centre aim point and a random background gray. Seeds are derived from the image name, so runs are reproducible.

Output boxes are 4-point quads (a rectangle stops being a rectangle under perspective), clockwise from top-left. Each box also has `visible_fraction`, the share of the box still inside the photo. Each entry stores the camera (`K`, `R`, `t`, distance, angles) and `H`, so a view can be reproduced or inverted.

Tests: `python -m pytest tests/test_camera.py`.

## Not done yet

- 3D scene with occluders (beetle, pin, other labels) so different text is hidden from different angles. Plan: render the label as a textured quad with `pyrender` or Open3D using the same `K`, `R`, `t`, and take occlusion from the depth buffer.
- Lens distortion, blur, lighting and noise, and paper curl (the label is perfectly flat here).
