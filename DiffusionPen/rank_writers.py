"""Generate probe words for many writers, score them, and write a ranking + contact sheet.
Run from the DiffusionPen folder:  python -u rank_writers.py --start 0 --end 40
"""
import argparse, os, subprocess, sys
import cv2, numpy as np
from label_generator import _get_verifier, _norm, _edit_distance, HF_SD_REPO

WORDS = ["Carabus", "Agonum", "Stenelmis", "Bledius"]

p = argparse.ArgumentParser()
p.add_argument("--start", type=int, default=0)
p.add_argument("--end", type=int, default=40)          # exclusive
args = p.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
samples = os.path.join(here, "image_samples")
read = _get_verifier()

rows, results = [], []
for w in range(args.start, args.end):
    paths = [os.path.join(samples, f"{x}_style_{w}.png") for x in WORDS]
    if not all(os.path.exists(q) for q in paths):      # skip generation if already done
        cmd = [sys.executable, "sample_names.py", "--train_mode", "sampling",
               "--sampling_mode", "single_sampling",
               "--save_path", "./diffusionpen_iam_model_path",
               "--style_path", "./style_models/iam_style_diffusionpen.pth",
               "--stable_dif_path", HF_SD_REPO,
               "--text", " ".join(WORDS), "--style_id", str(w), "--style_seed", "0"]
        r = subprocess.run(cmd, cwd=here, capture_output=True)
        if r.returncode or not all(os.path.exists(q) for q in paths):
            print(f"writer {w}: generation failed, skipping", flush=True)
            continue
    nlls = [read.nll(q, x) for q, x in zip(paths, WORDS)]
    dists = [_edit_distance(_norm(read(q)), _norm(x)) for q, x in zip(paths, WORDS)]
    score = float(np.mean(nlls))
    results.append((score, w, float(np.mean(dists))))
    print(f"writer {w}: mean nll {score:.2f}, mean edit dist {np.mean(dists):.1f}", flush=True)

    strip = [cv2.resize(cv2.imread(q), (256, 64)) for q in paths]
    row = np.hstack(strip)
    cv2.putText(row, f"w{w} nll {score:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
    rows.append((score, row))

rows.sort(key=lambda t: t[0])
if rows:
    cv2.imwrite(os.path.join(here, f"writers_{args.start}_{args.end}.png"), np.vstack([r for _, r in rows]))
print("\nBest writers (lowest nll):")
for score, w, d in sorted(results)[:15]:
    print(f"  writer {w}: nll {score:.2f}, edit dist {d:.1f}")