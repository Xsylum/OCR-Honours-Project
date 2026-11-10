import cv2
import numpy as np

from ProjectiveProjection.camera import Camera, box_to_quad, project_label, render_view
from ProjectiveProjection.lens import Lens
from ProjectiveProjection.occluders import Pin, StackedLabel, sample_stack
from ProjectiveProjection.sleeve import Sleeve

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


def stacked(z, width_mm=22.0, source="printed"):
    return StackedLabel(z_mm=z, width_mm=width_mm, angle_deg=0.0, pin_uv=(0.5, 0.5), source=source)


def test_label_stacked_above_hides_the_text():
    cam = Camera(distance_mm=60, tilt_deg=10)
    pin = Pin(pierce_mm=(0.0, 0.0), length_mm=10)
    _, regions, info, occ = render_view(LABEL, [NEAR_PIN, FAR], cam, np.random.default_rng(0), 22.0, False,
                                        pin, [stacked(-4.0, width_mm=30.0)])
    assert all(r["occluded_fraction"] > 0.9 for r in regions)
    assert [o["type"] for o in info["occluders"]] == ["pin", "stacked_label"]


def test_label_stacked_below_only_shows_around_the_edges():
    cam = Camera(distance_mm=60, tilt_deg=10)
    pin = Pin(pierce_mm=(0.0, 0.0), length_mm=10, below_mm=8)
    args = (LABEL, [NEAR_PIN, FAR], cam, np.random.default_rng(0), 22.0, False, pin)
    plain, plain_regions, _, plain_occ = render_view(*args)
    photo, regions, _, occ = render_view(*args, [stacked(4.0, width_mm=40.0)])
    assert np.array_equal(occ, plain_occ)
    assert [r["visible_fraction"] for r in regions] == [r["visible_fraction"] for r in plain_regions]
    assert not np.array_equal(photo, plain)


def test_sample_stack_extends_the_pin_through_every_label():
    rng = np.random.default_rng(1)
    for _ in range(20):
        pin = Pin(pierce_mm=(0.0, 0.0))
        stack = sample_stack(rng, pin, (22.0, 5.5), ["other.png"], max_above_mm=15.0)
        assert len(stack) <= 3
        for s in stack:
            assert -pin.length_mm < s.z_mm < pin.below_mm and s.z_mm >= -15.0


def test_lens_distortion_moves_boxes_with_the_ink():
    label = np.full((100, 400, 3), 255, np.uint8)
    label[10:40, 340:390] = 0  # ink exactly filling the FAR box, near the edge where distortion is strongest
    cam = Camera(distance_mm=40, tilt_deg=20, azimuth_deg=10)
    lens = Lens(k1=-0.4, k2=0.05, p1=0.002)
    photo, regions, info, _ = render_view(label, [NEAR_PIN, FAR], cam, np.random.default_rng(0), 22.0, False,
                                          lens=lens)
    ink = np.argwhere(photo[..., 0] < 50)[:, ::-1].mean(axis=0)
    _, plain, _, _ = render_view(label, [NEAR_PIN, FAR], cam, np.random.default_rng(0), 22.0, False)
    far, far_plain = np.mean(regions[1]["quad"], axis=0), np.mean(plain[1]["quad"], axis=0)
    assert np.linalg.norm(far - far_plain) > 5  # distortion is not negligible here...
    assert np.linalg.norm(far - ink) < 1.5  # ...and the box follows the ink
    restored = cv2.undistort(photo, np.array(info["camera"]["K"]), np.array(info["camera"]["distortion"]["opencv_dist_coeffs"]))
    restored_ink = np.argwhere(restored[..., 0] < 50)[:, ::-1].mean(axis=0)
    assert np.linalg.norm(restored_ink - far_plain) < 1.5


def aimed_sleeve(cam, point_mm, **kw):
    """A sleeve whose glare light mirrors into the camera at point_mm on the film."""
    s = Sleeve(**kw)
    C = cam.pose()[2]
    v = C - np.array([*point_mm, s.z_mm])
    v /= np.linalg.norm(v)
    s.glare_light = (v[0], v[1], -v[2])
    return s


def test_sleeve_glare_follows_the_mirror_direction_and_does_not_occlude():
    cam = Camera(distance_mm=60, tilt_deg=25)
    sleeve = aimed_sleeve(cam, (0.0, 0.0), gloss=0.9, roughness_deg=3.0, wrinkles=0)
    photo, regions, info, occ = render_view(LABEL, [NEAR_PIN, FAR], cam, np.random.default_rng(0), 22.0, False,
                                            sleeve=sleeve)
    near, far = regions
    assert near["glare_fraction"] > 0.5 and far["glare_fraction"] < 0.1
    assert near["occluded_fraction"] == 0 and not occ.any()
    assert info["sleeve"]["type"] == "sleeve"


def test_sleeve_lowers_text_contrast():
    label = LABEL.copy()
    label[30:70, 180:220] = 20
    cam = Camera(distance_mm=60)
    sleeve = aimed_sleeve(cam, (50.0, 50.0), haze=0.08, wrinkles=0, scratches=0)  # glare far off the label
    plain, *_ = render_view(label, [], cam, np.random.default_rng(0), 22.0, False)
    sleeved, *_ = render_view(label, [], cam, np.random.default_rng(0), 22.0, False, sleeve=sleeve)
    ink = lambda img: img[350:418, 478:546].astype(float).mean()
    paper = lambda img: img[300:330, 300:400].astype(float).mean()
    assert 0.6 < (paper(sleeved) - ink(sleeved)) / (paper(plain) - ink(plain)) < 0.97


def test_fogged_sleeve_blurs_the_text():
    label = LABEL.copy()
    label[20:80:6] = 20  # fine stripes across the whole label, wherever the fog patches fall
    cam = Camera(distance_mm=60)
    sharpness = lambda img: np.abs(cv2.Laplacian(img[..., 0].astype(np.float32), cv2.CV_32F)).mean()
    kw = dict(haze=0.0, gloss=0.0, wrinkles=0, scratches=0, scuffs=0, fog_blur_mm=0.15)
    clear, *_ = render_view(label, [], cam, np.random.default_rng(0), 22.0, False, sleeve=Sleeve(fog=0.0, **kw))
    fogged, *_ = render_view(label, [], cam, np.random.default_rng(0), 22.0, False, sleeve=Sleeve(fog=1.0, **kw))
    assert sharpness(fogged) < 0.8 * sharpness(clear)


def test_effects_are_reproducible_and_change_the_image():
    cam = Camera(distance_mm=50, tilt_deg=20)
    a, _, info, _ = render(cam, effects=True, seed=3)
    b, *_ = render(cam, effects=True, seed=3)
    clean, *_ = render(cam, effects=False)
    assert a.dtype == np.uint8 and a.shape == clean.shape
    assert np.array_equal(a, b)
    assert np.abs(a.astype(int) - clean).mean() > 2
    assert "jpeg_quality" in info["effects"]
