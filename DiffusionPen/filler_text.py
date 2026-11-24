import random
from PIL import Image, ImageDraw, ImageFont

LOREM = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et "
         "dolore magna aliqua enim ad minim veniam quis nostrud exercitation ullamco laboris nisi aliquip ex ea "
         "commodo consequat duis aute irure in reprehenderit voluptate velit esse cillum fugiat nulla pariatur "
         "excepteur sint occaecat cupidatat non proident sunt culpa qui officia deserunt mollit anim id est laborum").split()
FONTS = ["cour.ttf", "times.ttf", "arial.ttf", "DejaVuSansMono.ttf", "DejaVuSerif.ttf", "DejaVuSans.ttf"]
MIN_WIDTH = 420

def _font(size):
    for f in FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()

def add_filler_text(image_path, bg_rgb, margin=40, max_lines=3):
    """Adds 1..max_lines lines of lorem ipsum UNDER the name (so the name's boxes don't move).
    Overwrites image_path. Returns (extra_words [{"text","box"}], (W, H))."""
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    size = random.randint(14, 22)
    font = _font(size)
    n_lines = random.randint(1, max_lines)
    line_h = int(size * 1.5)
    W2, H2 = max(W, MIN_WIDTH), H + 8 + n_lines * line_h + margin // 2
    canvas = Image.new("RGB", (W2, H2), tuple(bg_rgb))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    ink = (random.randint(20, 60),) * 3
    extra, y = [], H + 8
    for _ in range(n_lines):
        x = margin
        for w in random.choices(LOREM, k=random.randint(2, 7)):
            x0, y0, x1, y1 = d.textbbox((x, y), w, font=font)
            if x1 > W2 - margin:
                break
            d.text((x, y), w, font=font, fill=ink)
            extra.append({"text": w, "box": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]})
            x = x1 + int(size * 0.4)
        y += line_h
    canvas.save(image_path)
    return extra, (W2, H2)