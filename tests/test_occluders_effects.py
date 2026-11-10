import numpy as np

from ProjectiveProjection.camera import Camera, box_to_quad, project_label, render_view
from ProjectiveProjection.occluders import Pin

LABEL = np.full((100, 400, 3), 240, np.uint8)
NEAR_PIN = {"text": "near", "quad": box_to_quad([180, 30, 40, 40])}  # label centre is (200, 50)
FAR = {"text": "far", "quad": box_to_quad([340, 10, 50, 30])}


def render(cam, pin=None, effects=False, seed=0):
    return render_view(LABEL, [NEAR_PIN, FAR], cam, np.random.default_rng(seed), 22.0, effects, pin)


def test_pin_occludes_text_at_the_pierce_point_only():
    # Tilting along the label's short axis makes the pin lean vertically in the photo, away from FAR.
    cam = Camera(distance_mm=60, tilt_deg=30, azimuth_deg=90)
    _, regions, info, occ = render(cam, Pin(pierce_mm=(0.0, 0.0)))
    near, far = regions
    assert near["occluded_fraction"] > 0.05 and near["visible_fraction"] < 1.0
    assert far["occluded_fraction"] == 0.0 and far["visible_fraction"] == 1.0
    assert occ.any() and info["occluders"][0]["type"] == "pin"


def test_oblique_view_shows_more_of_the_pin():
    pin = Pin(pierce_mm=(0.0, 0.0), length_mm=10)
    *_, top_down = render(Camera(distance_mm=60), pin)
    *_, oblique = render(Camera(distance_mm=60, tilt_deg=45), pin)
    extent = lambda m: np.ptp(np.argwhere(m), axis=0).max()
    assert extent(oblique) > 1.5 * extent(top_down)


def test_pin_reaching_past_the_camera_is_clipped():
    photo, *_ = render(Camera(distance_mm=40), Pin(pierce_mm=(0.0, 0.0), length_mm=500))
    assert photo.shape == (768, 1024, 3)


def test_clean_render_matches_pure_geometry():
    cam = Camera(distance_mm=50, tilt_deg=20, azimuth_deg=30, roll_deg=5)
    _, regions, _, occ = render(cam)
    _, geometric, _ = project_label(LABEL, [NEAR_PIN, FAR], cam, 22.0)
    assert [r["quad"] for r in regions] == [g["quad"] for g in geometric]
    assert not occ.any()


def test_effects_are_reproducible_and_change_the_image():
    cam = Camera(distance_mm=50, tilt_deg=20)
    a, _, info, _ = render(cam, effects=True, seed=3)
    b, *_ = render(cam, effects=True, seed=3)
    clean, *_ = render(cam, effects=False)
    assert a.dtype == np.uint8 and a.shape == clean.shape
    assert np.array_equal(a, b)
    assert np.abs(a.astype(int) - clean).mean() > 2
    assert "jpeg_quality" in info["effects"]
