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

GOOD_WRITERS = [21, 30, 6, 39, 36, 37, 25]
STYLE_SEED = 0
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
CANDIDATES_PER_SYLLABLE = 6     # max tries per syllable (stops early on an exact match). 1 = no retries
MAX_ACCEPT_DIST = 1        # edit distance allowed for strings longer than 4 letters
KEEP_CANDIDATES = True
MAX_NLL = 1.0
OCR_MODEL = "microsoft/trocr-base-handwritten"   # needs: pip install transformers torch   (downloaded once)
DEBUG_GUIDES = False            # True: also save <label>_guides.png with baseline (red), x-height line (blue), boxes
OVERLAP_SEARCH = 1.6    # how far back (x overlap width) to search the previous image for the matching letters
TARGET_CHAR_W = 27                      # px per letter, tune by eye
_OCR = {}
LAST_DISTS = {}
STATS_CSV = "writer_stats.csv"

def _norm(t):
    return "".join(c for c in t.lower() if c.isalnum())

def _log_stat(style, text, dist, ok):
    new = not os.path.exists(STATS_CSV)
    with open(STATS_CSV, "a") as f:
        if new:
            f.write("style,text,length,best_dist,accepted\n")
        f.write(f"{style},{text},{len(text)},{dist},{int(ok)}\n")

def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def _plausible(verifier, path, text, read):
    # a capitalised target must be read with a capital first letter
    if text[0].isupper() and not read.strip()[:1].isupper():
        return False
    # the target must beat every version of itself with one letter deleted
    base = verifier.logp(path, text)
    for i in range(len(text)):
        rival = text[:i] + text[i + 1:]
        if rival and rival != text and verifier.logp(path, rival) > base:
            return False
    return True

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

    def nll(path, text):
        """Mean NLL of `text` given the image, teacher-forced the same way generate() decodes."""
        img = Image.open(path).convert("RGB")
        with torch.no_grad():
            pv = proc(images=img, return_tensors="pt").pixel_values.to(dev)
            gen = model.generate(pv, max_new_tokens=24)[0]            # e.g. [2, 0, ...text..., 2]
            prefix = gen[:2].tolist()                                 # whatever generate itself starts with
            eos = model.config.decoder.eos_token_id
            body = proc.tokenizer(text, add_special_tokens=False).input_ids
            seq = torch.tensor([prefix + body + [eos]], device=dev)   # decoder input + targets in one sequence
            logits = model(pixel_values=pv, decoder_input_ids=seq[:, :-1]).logits
            logp = torch.log_softmax(logits, dim=-1)
            tgt = seq[:, 1:]
            tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            return -tok_lp[:, 1:].mean().item()                       # skip the position that predicts the 0/<s> token

    def logp(path, text):
        img = Image.open(path).convert("RGB")
        with torch.no_grad():
            pv = proc(images=img, return_tensors="pt").pixel_values.to(dev)
            gen = model.generate(pv, max_new_tokens=24)[0]
            prefix = gen[:2].tolist()
            eos = model.config.decoder.eos_token_id
            body = proc.tokenizer(text, add_special_tokens=False).input_ids
            seq = torch.tensor([prefix + body + [eos]], device=dev)
            logits = model(pixel_values=pv, decoder_input_ids=seq[:, :-1]).logits
            logp_ = torch.log_softmax(logits, dim=-1)
            tok_lp = logp_.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
            return tok_lp[:, 1:].sum().item()

    read.logp = logp


    read.nll = nll
    _OCR["fn"] = read
    return read


def generate_handwritten_syllables(data_item,  output_folder=None, style_id=None):
    """
    Renders every segment from the splitter with DiffusionPen and returns the file paths in order.
    A segment is rendered as overlap+core (e.g. 'na'+'cotar' -> 'nacotar'); the filename records both parts:
        syl_<idx>_<core>__<overlap>.png      (no '__' part when there is no overlap)
    """
    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "syllables")

    global STYLE_ID
    STYLE_ID = str(style_id if style_id is not None else random.choice(GOOD_WRITERS))

    global LAST_DISTS
    LAST_DISTS = {}

    segments = data_item.get("segments") or [
        {"generate": s, "core": s, "overlap": ""} for s in data_item["digits_or_syllables"]
    ]
    generated_image_paths = []

    diffusion_pen_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(diffusion_pen_dir, "sample_names.py")
    abs_style_path = os.path.abspath(os.path.join(diffusion_pen_dir, "style_models", "iam_style_diffusionpen.pth"))
    model_dir = os.path.join(diffusion_pen_dir, "diffusionpen_iam_model_path")
    samples_dir = os.path.join(diffusion_pen_dir, "image_samples")
    local_sd = os.path.join(diffusion_pen_dir, "stable-diffusion-v1-5", "stable-diffusion-v1-5")
    stable_dif_path = local_sd if os.path.isdir(local_sd) else HF_SD_REPO

    if os.name == 'nt':
        venv_python = os.path.join(diffusion_pen_dir, ".venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(diffusion_pen_dir, ".venv", "bin", "python")
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    os.makedirs(output_folder, exist_ok=True)
    abs_output_folder = os.path.abspath(output_folder)

    print(f"Triggering DiffusionPen generation for: {[s['generate'] for s in segments]}")

    verifier = _get_verifier() if VERIFY_WITH_OCR and CANDIDATES_PER_SYLLABLE > 1 else None
    attempts = CANDIDATES_PER_SYLLABLE if verifier else 1

    def run_once(text, dest):
        produced = os.path.join(samples_dir, f"{text}_style_{STYLE_ID}.png")
        if os.path.exists(produced):
            os.remove(produced)
        command = [
            venv_python, script_path,
            "--train_mode", "sampling",
            "--sampling_mode", "single_sampling",
            "--save_path", model_dir,
            "--style_path", abs_style_path,
            "--stable_dif_path", stable_dif_path,
            f"--text={text}",
            "--style_id", STYLE_ID,
            "--style_seed", str(STYLE_SEED),
        ]
        try:
            subprocess.run(command, cwd=diffusion_pen_dir, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
            print(f" DiffusionPen failed generating '{text}':\n{error_msg}")
            return False
        if not os.path.isfile(produced):
            print(f" DiffusionPen finished but {produced} was not created ('{text}')")
            return False
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        try:
            os.replace(produced, dest)
        except OSError as e:
            print(f" Could not move {produced} to {dest}: {e}")
            return False
        return True

    def clean(t):
        return "".join(c for c in t if c.isalnum())

    for idx, seg in enumerate(segments):
        gen_text = seg["generate"]                       # what the model is asked to draw (overlap + core)
        core, overlap = clean(seg["core"]), clean(seg.get("overlap", ""))
        name_part = f"{core}__{overlap}" if overlap else core
        file_path = os.path.join(abs_output_folder, f"syl_{idx}_{name_part}.png")


        allowed = 0 if len(gen_text) <= 5 else 1
        best_path, best_dist, best_score = None, None, None
        for attempt in range(attempts):
            cand = os.path.join(abs_output_folder, f"_cand_{idx}_{attempt}.png")
            if not run_once(gen_text, cand):
                continue
            if verifier is None:
                best_path = cand
                break
            read = verifier(cand)
            dist = _edit_distance(_norm(read), _norm(gen_text))
            # nll_val = verifier.nll(cand, gen_text)
            # score = (dist, nll_val)
            nll_val = 0.0
            score = (dist, 0.0)
            print(f"  attempt {attempt} for '{gen_text}' -> {read!r} (dist {dist}, nll {nll_val:.2f})")
            if KEEP_CANDIDATES:
                rej = os.path.join(abs_output_folder, "rejects")
                os.makedirs(rej, exist_ok=True)
                safe = "".join(c for c in read if c.isalnum()) or "blank"
                shutil.copy(cand, os.path.join(rej, f"w{STYLE_ID}_{gen_text}_{attempt}_dist{dist}_nll{nll_val:.2f}_read-{safe}.png"))
            if best_score is None or score < best_score:
                if best_path:
                    os.remove(best_path)
                best_path, best_dist, best_score = cand, dist, score
            else:
                os.remove(cand)
            if dist == 0 or (best_dist <= allowed and attempt >= 2):
                break

        if best_path is None:
            raise RuntimeError(f"no image generated for '{gen_text}'")
        if verifier is not None:
            _log_stat(STYLE_ID, gen_text, best_dist, best_dist <= allowed)
        if verifier is not None and best_dist > allowed:
            os.remove(best_path)
            raise RuntimeError(f"'{gen_text}' never read back correctly "
                               f"(best dist {best_dist}); skipping label")
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
        os.replace(best_path, file_path)
        generated_image_paths.append(file_path)
        LAST_DISTS[gen_text] = best_dist

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

    m = (alpha > 64).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 2:
        H_, W_ = alpha.shape
        big = st[1:, cv2.CC_STAT_AREA].max()
        comps = sorted(range(1, n), key=lambda i: st[i, cv2.CC_STAT_LEFT])
        last = comps[-1]
        prev_right = max(st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] for i in comps[:-1])
        gap = st[last, cv2.CC_STAT_LEFT] - prev_right
        small = st[last, cv2.CC_STAT_AREA] < 0.05 * big and st[last, cv2.CC_STAT_WIDTH] < 0.08 * W_
        low = st[last, cv2.CC_STAT_TOP] > 0.5 * H_      # a full stop sits near the baseline; an i-dot is high
        if small and low and gap >= 2:
            alpha[lab == last] = 0
    
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


def _col_profile(mask):
    """Ink amount per column."""
    return (mask > 40).sum(axis=0).astype(np.float32)


def _snap_to_valley(prof, x, tol):
    """Column near x with the least ink (the gap between letters), searched within +-tol."""
    lo, hi = max(1, int(x - tol)), min(len(prof) - 1, int(x + tol) + 1)
    if hi <= lo:
        return int(np.clip(x, 0, len(prof)))
    return lo + int(np.argmin(prof[lo:hi]))


def _align_overlap(prev_mask, new_mask, split):
    """x offset (in prev_mask coordinates) at which new_mask's left edge should sit so that its first `split`
    columns (the overlap letters) lie on top of the same letters at the end of prev_mask."""
    pp, npf = _col_profile(prev_mask), _col_profile(new_mask)
    L = min(len(pp), int(split * OVERLAP_SEARCH))
    if split < 4 or L < 4:
        return len(pp) - split                           # too little to correlate: core starts at prev's end
    head, tail = npf[:split], pp[-L:]
    best, best_off = -1e9, L - split
    for off in range(0, L):
        m = min(split, L - off)
        if m < split // 2:
            break
        a, b = head[:m], tail[off:off + m]
        score = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6) - 0.15 * (split - m) / split
        if score > best:
            best, best_off = score, off
    return len(pp) - L + best_off


def stitch_handwritten_words_to_solid(syllable_images, output_image_name, name=None):
    """Compose one label from segment images and return the OCR boxes [x, y, w, h] (one per core chunk).

    Segments with an overlap are lined up on the letters they share with the previous segment; their own copy
    of the overlap is then cropped away so the letters appear once. Boxes cover the core ink only."""
    glyphs = []
    for path in syllable_images:
        mask = _load_syllable_mask(path)
        if mask is None:
            continue
        m = re.match(r"syl_\d+_([^_]*)(?:__(.*))?\.png$", os.path.basename(path))
        core = m.group(1) if m else ""
        ov = (m.group(2) or "") if m else ""
        glyphs.append({"file": path, "text": core, "overlap": ov, "gen_text": ov + core, "mask": mask})
    if not glyphs:
        return []

    # 1. rescale each image to the common x-height (geometry from the FULL generated string, since that is
    #    what the image contains), then locate where the overlap ends inside it
    geo = [_geometry(g["gen_text"], g["mask"].shape[0]) for g in glyphs]
    scales = [TARGET_XHEIGHT / gm[0] if gm else None for gm in geo]
    known = [v for v in scales if v is not None]
    default_scale = float(np.median(known)) if known else 1.0
    raw = []
    for g, gm, sc in zip(glyphs, geo, scales):
        sc = default_scale if sc is None else sc
        w = g["mask"].shape[1]
        raw.append((sc * TARGET_CHAR_W * len(g["gen_text"]) / w) ** 0.5)
    mid = float(np.median(raw))
    final = [float(np.clip(s, 0.9 * mid, 1.1 * mid)) for s in raw]   # words in one label stay within ±10%
    for g, gm, sc in zip(glyphs, geo, final):
        h, w = g["mask"].shape
        interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(g["mask"], (max(1, round(w * sc)), max(1, round(h * sc))), interpolation=interp)
        base = (gm[1] if gm else h) * sc

        # overlap boundary: proportional guess by letter count, snapped to the nearest gap between strokes
        split = 0
        if g["overlap"]:
            guess = resized.shape[1] * len(g["overlap"]) / len(g["gen_text"])
            split = _snap_to_valley(_col_profile(resized), guess, 0.15 * resized.shape[1])

        angle = random.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
        jitter = random.randint(-MAX_JITTER_PX, MAX_JITTER_PX)
        rh, rw = resized.shape
        pad = int(0.25 * max(rh, rw)) + 4
        canvas = cv2.copyMakeBorder(resized, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        cx, cy = pad + rw / 2.0, pad + base
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ys, xs = np.where(rot > 40)
        if ys.size == 0:
            g["mask"] = None
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        new_mask = rot[y0:y1, x0:x1]
        split = int(np.clip(split + pad - x0, 0, max(0, new_mask.shape[1] - 4))) if split else 0
        g.update(mask=new_mask, split=split, base=int(round(cy - y0)), angle=angle, jitter=jitter, scale=sc)
    glyphs = [g for g in glyphs if g["mask"] is not None]
    if not glyphs:
        return []

    # 2. lay out left to right on one baseline.
    #    overlapped segment -> align on shared letters, crop its overlap, reuse the previous baseline wobble
    #    other segments     -> small gap inside a word, bigger gap between words
    breaks = _word_breaks([g["text"] for g in glyphs], name)
    above = max(g["base"] for g in glyphs) + MAX_JITTER_PX
    below = max(g["mask"].shape[0] - g["base"] for g in glyphs) + MAX_JITTER_PX
    base_y = FINAL_MARGIN + above
    x, placed = FINAL_MARGIN, []
    for i, g in enumerate(glyphs):
        if i > 0 and g["split"] > 0:
            prev, (prev_x, _) = glyphs[i - 1], placed[-1]
            new_x = prev_x + _align_overlap(prev["mask"], g["mask"], g["split"])
            g["mask"] = g["mask"][:, g["split"]:]                 # keep the previous image's copy of the overlap
            px = max(prev_x + 1, int(round(new_x + g["split"])))
            g["jitter"] = prev["jitter"]                          # stay on the same wobble so strokes connect
        else:
            if i > 0:
                x += random.randint(*(WORD_GAP if i in breaks else SYLLABLE_GAP))
            px = x
        y = base_y - g["base"] + g["jitter"]
        placed.append((px, y))
        x = px + g["mask"].shape[1]
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
            "overlap": g["overlap"],
            "box": [int(px), int(py), int(w), int(h)],      # core ink only: [x, y, w, h]
            "rotation_deg": g["angle"],
            "jitter_px": g["jitter"],
            "scale": g["scale"],
        })

    # 3. blend the ink onto the solid background colour
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    bg = np.full((H, W, 3), BACKGROUND_COLOR, np.float32)
    ink = np.full((H, W, 3), INK_COLOR, np.float32)
    final_label_bgr = (ink * a + bg * (1.0 - a)).astype(np.uint8)
    cv2.imwrite(output_image_name, final_label_bgr)

    if DEBUG_GUIDES:
        vis = final_label_bgr.copy()
        cv2.line(vis, (0, base_y), (W, base_y), (0, 0, 220), 1)
        cv2.line(vis, (0, base_y - TARGET_XHEIGHT), (W, base_y - TARGET_XHEIGHT), (220, 90, 0), 1)
        for b in word_boxes:
            bx, by, bw, bh = b["box"]
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 160, 0), 1)
        cv2.imwrite(os.path.splitext(output_image_name)[0] + "_guides.png", vis)

    return word_boxes