import json
import random

import cv2
import numpy as np

from ProjectiveProjection.camera import (Camera, box_to_quad, homography, label_to_world, project_dataset,
                                         project_label, sample_camera)


def test_fronto_parallel_camera_is_a_centred_similarity():
    cam = Camera(distance_mm=100, image_size=(800, 600))
    S = label_to_world(400, 100, mm_per_px=0.05)
    H = homography(cam, S)
    centre = cv2.perspectiveTransform(np.array([[[200.0, 50.0]]]), H)[0, 0]
    assert np.allclose(centre, [400, 300], atol=1e-6)
    assert np.allclose(H[2, :2], 0, atol=1e-9)  # no perspective term
    assert np.isclose(H[0, 0], H[1, 1]) and np.isclose(H[0, 1], 0, atol=1e-9)


def test_tilt_foreshortens():
    S = label_to_world(400, 100, 0.05)
    flat = homography(Camera(distance_mm=100), S)
    tilted = homography(Camera(distance_mm=100, tilt_deg=60, azimuth_deg=0), S)
    width = lambda H: np.ptp(cv2.perspectiveTransform(np.array([[[0.0, 50.0], [400.0, 50.0]]]), H)[0, :, 0])
    assert width(tilted) < width(flat) * 0.7


def test_quads_follow_the_pixels():
    img = np.full((100, 400, 3), 255, np.uint8)
    cv2.circle(img, (300, 30), 4, (0, 0, 0), -1)
    regions = [{"text": "x", "quad": box_to_quad([296, 26, 8, 8])}]
    for seed in range(5):
        cam = sample_camera(random.Random(seed), (22.0, 5.5), image_size=(800, 600))
        photo, warped, _ = project_label(img, regions, cam, label_width_mm=22.0, background=255)
        ys, xs = np.nonzero(photo[:, :, 0] < 128)
        dot = np.array([xs.mean(), ys.mean()])
        assert np.linalg.norm(dot - np.mean(warped[0]["quad"], axis=0)) < 3.0
        assert warped[0]["visible_fraction"] == 1.0


def test_reads_label_generator_format(tmp_path):
    img = np.full((120, 500, 3), 240, np.uint8)
    cv2.imwrite(str(tmp_path / "label_0.png"), img)
    master = [{"image_file": "label_0.png", "ground_truth": "Carabus auratus",
               "words": [{"text": "Cara", "box": [40, 40, 100, 50]}, {"text": "bus", "box": [150, 40, 80, 50]}]}]
    (tmp_path / "master.json").write_text(json.dumps(master))
    out = project_dataset(tmp_path / "master.json", tmp_path, tmp_path / "out", views=2)
    assert [v["image_file"] for v in out] == ["label_0_v0.png", "label_0_v1.png"]
    assert [w["text"] for w in out[0]["words"]] == ["Cara", "bus"]
    saved = json.loads((tmp_path / "out" / "projected_dataset_labels.json").read_text())
    assert len(saved) == 2 and (tmp_path / "out" / "label_0_v1.png").exists()
