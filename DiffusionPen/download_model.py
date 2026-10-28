"""Download the pretrained DiffusionPen weights that sample_names.py needs.

They come from the authors' Hugging Face repo (konnik/DiffusionPen), so nothing large lives in git.
Files that are already on disk are skipped.

    python download_models.py

or from code:
    from DiffusionPen.download_models import ensure_models
    ensure_models()
"""
import os

from huggingface_hub import hf_hub_download

REPO_ID = "konnik/DiffusionPen"

HERE = os.path.dirname(os.path.abspath(__file__))

# Only what sampling needs, laid out exactly as label_generator.py expects.
# (The repo also has optim.pt, ~1.35 GB, which is only used to resume training, so it is skipped.)
WEIGHT_FILES = [
    "style_models/iam_style_diffusionpen.pth",
    "diffusionpen_iam_model_path/models/ckpt.pt",
    "diffusionpen_iam_model_path/models/ema_ckpt.pt",
]


def ensure_models():
    """Download any weight file that isn't on disk yet. Does nothing if they're all present."""
    for rel in WEIGHT_FILES:
        if os.path.isfile(os.path.join(HERE, rel)):
            continue
        print(f"Downloading {rel} from {REPO_ID} ...")
        hf_hub_download(repo_id=REPO_ID, filename=rel, local_dir=HERE)


if __name__ == "__main__":
    ensure_models()