#label_generator.py
import cv2
import numpy as np
import json
import random
import re
import shutil
import subprocess
import os
import sys

STYLE_ID = "1"
# Hugging Face repo id; downloaded once into the HF cache, then reused. If you ever put a full copy of the
# model in DiffusionPen/stable-diffusion-v1-5/stable-diffusion-v1-5 it is used instead.
HF_SD_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# ---------------------------------------------------------------- label look (pixels) -- tune these
TARGET_XHEIGHT = 40             # height of lowercase letters like a, e, o. Ascenders reach ~2x, descenders go ~0.9x below
# Handwriting proportions, in x-heights, measured on real DiffusionPen output (IAM style). They are used to work
# out each syllable's x-height from the letters in it. If syllables still look too big/small depending on
# their letters, nudge these (see DEBUG_GUIDES).
ASC_RATIO = 0.5                 # lowercase b d f h k l rise this far above the x-height line (short in this style)
CAP_RATIO = 1.0                 # capitals and digits rise this far above the x-height line (about 2x x-height tall)
DESC_RATIO = 0.7                # g j p q y reach this far below the baseline
DOT_RATIO = 0.3                 # dot of i / j above the x-height line
SYLLABLE_GAP = (0, 8)           # random gap (px) between syllables of the same word
WORD_GAP = (30, 50)             # random gap (px) between words (needs the `name` argument, see below)
MAX_ROTATION_DEG = 1.5          # tiny per-syllable tilt
MAX_JITTER_PX = 3               # tiny per-syllable vertical wobble
FINAL_MARGIN = 40
BACKGROUND_COLOR = (245, 245, 240)   # BGR
INK_COLOR = (25, 25, 25)             # BGR
# --- pick the best of several tries per syllable (DiffusionPen draws some syllables wrongly, e.g. t -> l, x -> k)
VERIFY_WITH_OCR = True          # generate several candidates and keep the one a handwriting-OCR model reads best
CANDIDATES_PER_SYLLABLE = 4     # max tries per syllable (stops early on an exact match). 1 = no retries
OCR_MODEL = "microsoft/trocr-small-handwritten"   # needs: pip install transformers torch   (downloaded once)
DEBUG_GUIDES = False            # True: also save <label>_guides.png with baseline (red), x-height line (blue), boxes


_OCR = {}


def _norm(t):
    return "".join(c for c in t.lower() if c.isalnum())


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _get_verifier():
    """Returns read(path) -> text using a pretrained handwriting OCR model (loaded once), or None if unavailable."""
    if "fn" in _OCR:
        return _OCR["fn"]
    try:
        import torch
        from PIL import Image
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError:
        print("OCR verification is off: run  pip install transformers torch  (in the environment that runs pipeline.py)")
        _OCR["fn"] = None
        return None
    try:
        proc = TrOCRProcessor.from_pretrained(OCR_MODEL)
        model = VisionEncoderDecoderModel.from_pretrained(OCR_MODEL).eval()
    except Exception as e:
        print(f"OCR verification is off (could not load {OCR_MODEL}): {e}\n"
              "Try:  pip install sentencepiece protobuf   or set OCR_MODEL = 'microsoft/trocr-base-handwritten'")
        _OCR["fn"] = None
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    def read(path):
        img = Image.open(path).convert("RGB")
        with torch.no_grad():
            pv = proc(images=img, return_tensors="pt").pixel_values.to(dev)
            ids = model.generate(pv, max_new_tokens=24)
        return proc.batch_decode(ids, skip_special_tokens=True)[0]

    _OCR["fn"] = read
    return read


def generate_handwritten_syllables(data_item, output_folder="./output_labels"):
    """
    Accepts the dictionary from the syllable splitter, feeds each syllable
    individually into the DiffusionPen model via subprocess CLI, and returns
    the list of newly generated file paths.
    """
    syllables = data_item["digits_or_syllables"]  # e.g., ['Pi', 'na', 'co', 'tar', 'sus']
    generated_image_paths = []

    diffusion_pen_dir = os.path.dirname(os.path.abspath(__file__))

    script_path = os.path.join(diffusion_pen_dir, "sample_names.py")

    abs_style_path = os.path.abspath(os.path.join(diffusion_pen_dir, "style_models", "iam_style_diffusionpen.pth"))

    # --save_path in sample_names.py is the MODEL folder (it loads models/ckpt.pt and models/ema_ckpt.pt from it),
    # not an output file.
    model_dir = os.path.join(diffusion_pen_dir, "diffusionpen_iam_model_path")

    # sample_names.py writes its result to ./image_samples/<text>_style_<id>.png (relative to its working dir)
    samples_dir = os.path.join(diffusion_pen_dir, "image_samples")

    # diffusers wants a plain folder path or a Hugging Face repo id -- NOT a file:// URI
    local_sd = os.path.join(diffusion_pen_dir, "stable-diffusion-v1-5", "stable-diffusion-v1-5")
    stable_dif_path = local_sd if os.path.isdir(local_sd) else HF_SD_REPO

    if os.name == 'nt':  # Windows
        venv_python = os.path.join(diffusion_pen_dir, ".venv", "Scripts", "python.exe")
    else:  # Mac / Linux
        venv_python = os.path.join(diffusion_pen_dir, ".venv", "bin", "python")

    if not os.path.exists(venv_python):
        venv_python = sys.executable

    # Ensure your output folder exists
    os.makedirs(output_folder, exist_ok=True)
    abs_output_folder = os.path.abspath(output_folder)

    print(f"Triggering DiffusionPen generation for syllables: {syllables}")

    verifier = _get_verifier() if VERIFY_WITH_OCR and CANDIDATES_PER_SYLLABLE > 1 else None
    attempts = CANDIDATES_PER_SYLLABLE if verifier else 1

    def run_once(syllable, dest):
        """Run DiffusionPen once for this syllable and move the result to `dest`. Returns True on success."""
        produced = os.path.join(samples_dir, f"{syllable}_style_{STYLE_ID}.png")
        if os.path.exists(produced):
            os.remove(produced)                      # never pick up a stale image
        command = [
            venv_python, script_path,
            "--train_mode", "sampling",
            "--sampling_mode", "single_sampling",
            "--save_path", model_dir,
            "--style_path", abs_style_path,
            "--stable_dif_path", stable_dif_path,
            f"--text={syllable}",                    # '=' form so a syllable like '-' is not parsed as an option
            "--style_id", STYLE_ID,
        ]
        try:
            subprocess.run(command, cwd=diffusion_pen_dir, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
            print(f" DiffusionPen failed generating syllable '{syllable}':\n{error_msg}")
            return False
        if not os.path.isfile(produced):
            print(f" DiffusionPen finished but {produced} was not created (syllable '{syllable}')")
            return False
        if os.path.isdir(dest):                      # leftover folder from an old failed run ([WinError 5])
            shutil.rmtree(dest)
        try:
            os.replace(produced, dest)
        except OSError as e:
            print(f" Could not move {produced} to {dest}: {e}")
            return False
        return True

    for idx, syllable in enumerate(syllables):
        sanitized_text = "".join(c for c in syllable if c.isalnum())
        file_path = os.path.join(abs_output_folder, f"syl_{idx}_{sanitized_text}.png")

        best_path, best_dist = None, None
        for attempt in range(attempts):
            cand = os.path.join(abs_output_folder, f"_cand_{idx}_{attempt}.png")
            if not run_once(syllable, cand):
                continue
            if verifier is None:
                best_path = cand
                break
            dist = _edit_distance(_norm(verifier(cand)), _norm(syllable))
            if best_dist is None or dist < best_dist:
                if best_path:
                    os.remove(best_path)
                best_path, best_dist = cand, dist
            else:
                os.remove(cand)
            if dist == 0:
                break
        if best_path is None:
            continue
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
        os.replace(best_path, file_path)
        if best_dist:
            print(f" '{syllable}': no exact OCR match after {attempts} tries, kept the closest (edit distance {best_dist})")
        generated_image_paths.append(file_path)

    # Returns the list of image file paths so your OpenCV stitching function can immediately open them
    return generated_image_paths


_TALL = set('bdfhkl')
_DESC = set('gjpqy')


def _ink_mask_and_box(path):
    """DiffusionPen saves an OPAQUE 3-channel image: dark ink on a slightly noisy near-white 64x256 background.
    Returns (alpha mask uint8 with the background removed, (x0, y0, x1, y1) box of the ink) or (None, None)."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, None
    ink = 255.0 - gray.astype(np.float32)
    floor = float(np.clip(np.median(ink) + 25.0, 15.0, 90.0))      # background noise level
    strong = ink[ink > floor]
    if strong.size == 0:
        return None, None
    ref = max(float(np.percentile(strong, 98)), floor + 30.0)       # darkest strokes -> full opacity
    alpha = (np.clip((ink - floor) / (ref - floor), 0.0, 1.0) * 255.0).astype(np.uint8)
    on = alpha > 64
    cols = np.where(on.sum(axis=0) >= 2)[0]
    rows = np.where(on.sum(axis=1) >= 2)[0]
    if cols.size == 0 or rows.size == 0:
        return None, None
    return alpha, (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def _load_syllable_mask(path):
    alpha, box = _ink_mask_and_box(path)
    if alpha is None:
        return None
    x0, y0, x1, y1 = box
    return alpha[y0:y1, x0:x1]


def _geometry(text, height):
    """(x-height, baseline row) in the syllable image's own pixels, from its ink height and its letters.

    DiffusionPen scales every word so the ink fills ~64 px of height, so a syllable without tall letters ('ra')
    is drawn far bigger than one with ascenders/descenders ('gy', 'bli'). The ink height is
        x-height * (1 + ascender part + descender part)
    so x-height = height / that ratio, and the baseline is one descender above the bottom of the ink.
    Returns None for punctuation-only text."""
    letters = [c for c in text if c.isalnum()]
    if not letters:
        return None
    up = 0.0
    if any(c.isupper() or c.isdigit() for c in letters):
        up = CAP_RATIO
    if any(c in _TALL for c in letters):
        up = max(up, ASC_RATIO)
    if 't' in letters:
        up = max(up, 0.6 * ASC_RATIO)             # t is only a short ascender
    if any(c in 'ij' for c in letters):
        up = max(up, DOT_RATIO)
    down = DESC_RATIO if any(c in _DESC for c in letters) else 0.0
    xh = height / (1.0 + up + down)
    return xh, height - down * xh


def _word_breaks(texts, name):
    """Indices of syllables that start a new word, found by matching the syllables against the name's words.
    Returns an empty set when there is no name or the syllables don't spell it (e.g. one failed to generate)."""
    if not name:
        return set()
    words = [''.join(c for c in w.lower() if c.isalnum()) for w in name.split()]
    breaks, i = set(), 0
    for wi, w in enumerate([w for w in words if w]):
        start, acc = i, ''
        while i < len(texts) and len(acc) < len(w):
            acc += texts[i].lower()
            i += 1
        if acc != w:
            return set()
        if wi > 0:
            breaks.add(start)
    return breaks if i == len(texts) else set()


def stitch_handwritten_words_to_solid(syllable_images, output_image_name, name=None):
    """Compose one label from syllable images and return the OCR boxes [x, y, w, h] (one per syllable).

    Every syllable is rescaled to the same x-height and placed on one shared baseline, so the result reads like
    a single piece of handwriting. Pass `name` (e.g. 'Carabus auratus') to get a larger gap between words."""
    glyphs = []
    for path in syllable_images:
        mask = _load_syllable_mask(path)
        if mask is None:
            continue
        m = re.match(r"syl_\d+_(.*)\.png$", os.path.basename(path))
        glyphs.append({"file": path, "text": m.group(1) if m else "", "mask": mask})
    if not glyphs:
        return []

    # 1. rescale each syllable to the common x-height and find its baseline
    geo = [_geometry(g["text"], g["mask"].shape[0]) for g in glyphs]
    scales = [TARGET_XHEIGHT / gm[0] if gm else None for gm in geo]
    known = [v for v in scales if v is not None]
    default_scale = float(np.median(known)) if known else 1.0          # punctuation takes the typical scale
    for g, gm, sc in zip(glyphs, geo, scales):
        sc = default_scale if sc is None else sc
        h, w = g["mask"].shape
        interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(g["mask"], (max(1, round(w * sc)), max(1, round(h * sc))), interpolation=interp)
        base = (gm[1] if gm else h) * sc

        # 2. tiny random tilt around the baseline point + tiny vertical wobble (kept small so it still looks like
        #    one line of writing)
        angle = random.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
        jitter = random.randint(-MAX_JITTER_PX, MAX_JITTER_PX)
        rh, rw = resized.shape
        pad = int(0.25 * max(rh, rw)) + 4
        canvas = cv2.copyMakeBorder(resized, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        cx, cy = pad + rw / 2.0, pad + base
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)               # the baseline point stays where it is
        rot = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ys, xs = np.where(rot > 40)
        if ys.size == 0:
            g["mask"] = None
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        g.update(mask=rot[y0:y1, x0:x1], base=int(round(cy - y0)), angle=angle, jitter=jitter, scale=sc)
    glyphs = [g for g in glyphs if g["mask"] is not None]
    if not glyphs:
        return []

    # 3. lay out left to right on one baseline: small gaps inside a word, a bigger gap between words
    breaks = _word_breaks([g["text"] for g in glyphs], name)
    above = max(g["base"] for g in glyphs) + MAX_JITTER_PX
    below = max(g["mask"].shape[0] - g["base"] for g in glyphs) + MAX_JITTER_PX
    base_y = FINAL_MARGIN + above
    x, placed = FINAL_MARGIN, []
    for i, g in enumerate(glyphs):
        if i > 0:
            x += random.randint(*(WORD_GAP if i in breaks else SYLLABLE_GAP))
        y = base_y - g["base"] + g["jitter"]
        placed.append((x, y))
        x += g["mask"].shape[1]
    W, H = x + FINAL_MARGIN, base_y + below + FINAL_MARGIN

    alpha = np.zeros((H, W), np.uint8)
    word_boxes = []
    for g, (px, py) in zip(glyphs, placed):
        h, w = g["mask"].shape
        region = alpha[py:py + h, px:px + w]
        np.maximum(region, g["mask"], out=region)
        word_boxes.append({
            "file": g["file"],
            "text": g["text"],
            "box": [int(px), int(py), int(w), int(h)],      # tight box around this syllable's ink: [x, y, w, h]
            "rotation_deg": g["angle"],
            "jitter_px": g["jitter"],
            "scale": g["scale"],
        })

    # 4. blend the ink onto the solid background colour
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    bg = np.full((H, W, 3), BACKGROUND_COLOR, np.float32)
    ink = np.full((H, W, 3), INK_COLOR, np.float32)
    final_label_bgr = (ink * a + bg * (1.0 - a)).astype(np.uint8)
    cv2.imwrite(output_image_name, final_label_bgr)

    if DEBUG_GUIDES:
        vis = final_label_bgr.copy()
        cv2.line(vis, (0, base_y), (W, base_y), (0, 0, 220), 1)                              # baseline
        cv2.line(vis, (0, base_y - TARGET_XHEIGHT), (W, base_y - TARGET_XHEIGHT), (220, 90, 0), 1)   # x-height line
        for b in word_boxes:
            bx, by, bw, bh = b["box"]
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 160, 0), 1)
        cv2.imwrite(os.path.splitext(output_image_name)[0] + "_guides.png", vis)

    return word_boxes