# label_generation_pipeline.py
import json
import sys
import os
import random
import numpy as np
import cv2

from CatalogueOfLife.extract_beetles import get_names_list
from DiffusionPen.syllable_splitter import split_name_into_syllables
from DiffusionPen.label_generator import generate_handwritten_syllables, stitch_handwritten_words_to_solid
from DiffusionPen.filler_text import add_filler_text
import DiffusionPen.label_generator as lg

OUTPUT_DIR = "./final_output_labels"
PREVIEW_DIR = os.path.join(OUTPUT_DIR, "preview")
CAPTION_HEIGHT = 36
LABELS_PER_NAME = 2
NUM_NAMES = 20      # raise for the real run
IDS_DIR = os.path.join(OUTPUT_DIR, "ids")

def group_into_words(name, syllable_boxes):
    """Merge per-chunk boxes into one box per word of `name`.
    Returns [{"text", "box"}] or None if the chunks don't spell the name
    (e.g. a chunk image was dropped during stitching)."""
    words, i, out = name.split(), 0, []
    for w in words:
        acc, group = "", []
        while i < len(syllable_boxes) and len(acc) < len(w):
            acc += syllable_boxes[i]["text"]
            group.append(syllable_boxes[i])
            i += 1
        if acc.lower() != w.lower():
            return None
        x0 = min(b["box"][0] for b in group)
        y0 = min(b["box"][1] for b in group)
        x1 = max(b["box"][0] + b["box"][2] for b in group)
        y1 = max(b["box"][1] + b["box"][3] for b in group)
        out.append({"text": w, "box": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]})
    return out if i == len(syllable_boxes) else None


def save_preview(image_path, preview_path, name, words, syllables):
    """Copy of the label with a caption strip, word boxes (green) and chunk boxes (thin blue)."""
    img = cv2.imread(image_path)
    if img is None:
        return
    for s in syllables:
        x, y, w, h = s["box"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 120, 0), 1)
    for wd in words:
        x, y, w, h = wd["box"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 160, 0), 2)
        cv2.putText(img, wd["text"], (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 200), 1, cv2.LINE_AA)
    band = img[:1].copy()
    band = cv2.resize(band, (img.shape[1], CAPTION_HEIGHT))
    band[:] = (255, 255, 255)
    cv2.putText(band, name, (8, CAPTION_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(preview_path, cv2.vconcat([band, img]))

def box_to_poly(box, n=4):
    """[x,y,w,h] -> clockwise polygon, n points per edge, starting top-left (corners are points 0, n, 2n, 3n)."""
    x, y, w, h = box
    xs = [x + w * i / n for i in range(n)]
    ys = [y + h * i / n for i in range(n)]
    pts = [(px, y) for px in xs] + [(x + w, py) for py in ys] \
        + [(x + w - w * i / n, y + h) for i in range(n)] + [(x, y + h - h * i / n) for i in range(n)]
    return [[round(float(a), 2), round(float(b), 2)] for a, b in pts]

def write_id_mask(path, size, items):
    """uint16 PNG, same size as the label: 0 = background, k = items[k-1]'s box."""
    W, H = size
    ids = np.zeros((H, W), np.uint16)
    for k, it in enumerate(items, 1):
        x, y, w, h = it["box"]
        ids[y:y + h, x:x + w] = k
    cv2.imwrite(path, ids)

def run_dataset_pipeline(shard=0, n_shards=1):
    print(f"Starting dataset generation (shard {shard}/{n_shards})")
    names = get_names_list(limit=None)
    random.Random(0).shuffle(names)              # same order in every shard
    raw_names = names[:NUM_NAMES]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    os.makedirs(IDS_DIR, exist_ok=True)
    syl_dir = f"./syllables_{shard}"
    jsonl_path = f"labels_shard{shard}.jsonl"

    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            done = {json.loads(l)["index"] for l in f if l.strip()}

    n_labels = names_covered = 0
    for index in range(shard, len(raw_names), n_shards):
        if index in done:
            continue
        name = raw_names[index]
        print(f"\n--- [{index + 1}/{len(raw_names)}]: {name} ---")
        split_data = split_name_into_syllables(name)
        slug = "_".join(name.split())
        made = 0

        for writer in random.sample(lg.GOOD_WRITERS, len(lg.GOOD_WRITERS))[:4]:
            if made >= LABELS_PER_NAME:
                break
            try:
                syllable_image_paths = generate_handwritten_syllables(split_data, output_folder=syl_dir, style_id=writer)
            except Exception as e:
                print(f"  writer {writer} failed for '{name}': {e}")
                continue
            if not syllable_image_paths:
                continue

            style_id = lg.STYLE_ID
            image_name = f"label_{index}_{slug}_w{style_id}.png"
            final_image_path = os.path.join(OUTPUT_DIR, image_name)

            syllable_boxes = stitch_handwritten_words_to_solid(syllable_image_paths, final_image_path, name=name)
            if not syllable_boxes or not os.path.isfile(final_image_path):
                continue
            words = group_into_words(name, syllable_boxes)
            if words is None:
                os.remove(final_image_path)
                continue

            # filler text goes under the name; this is where filler_text.py is called
            extra, size = add_filler_text(final_image_path, lg.BACKGROUND_COLOR[::-1])
            items = words + extra
            for it in items:
                it["poly"] = box_to_poly(it["box"])
            ids_name = image_name.replace(".png", "_ids.png")
            write_id_mask(os.path.join(IDS_DIR, ids_name), size, items)

            syllables = [{"text": b["text"], "box": b["box"]} for b in syllable_boxes]
            save_preview(final_image_path, os.path.join(PREVIEW_DIR, image_name),
                         f"{name} [writer {style_id}]", items, syllables)

            entry = {
                "index": index,
                "image_file": image_name,
                "image_size": list(size),
                "ground_truth": name,
                "words": words,              # the beetle name: box + poly per word (what camera.py reads)
                "extra_words": extra,        # lorem ipsum filler, same format
                "id_mask": f"ids/{ids_name}",
                "syllables": syllables,
                "style_id": style_id,
                "ocr_dist": dict(lg.LAST_DISTS),
            }
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            made += 1
            n_labels += 1

        names_covered += (made > 0)

    print(f"Shard {shard} done: {n_labels} labels covering {names_covered} names.")


if __name__ == "__main__":
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_shards = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run_dataset_pipeline(shard, n_shards)