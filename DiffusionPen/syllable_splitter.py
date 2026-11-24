# syllable_splitter.py
"""
Splits a name into word-like chunks for DiffusionPen, with optional overlap for cursive stitching.

Each word is cut into "core" chunks (3-6 letters, 4-5 preferred) at natural-looking points.
Every core after the first in a word is *generated* with the tail of the previous core
prepended (the overlap), so neighbouring images both contain the same letters and the
stitcher can align on them.

Return shape:
{
  "original_name": "Pinacotarsus proximus",
  "digits_or_syllables": ["Pina", "cotar", "sus", "proxi", "mus"],   # cores only (old key, unchanged meaning)
  "segments": [
     {"generate": "Pina",   "core": "Pina",  "overlap": "",   "word_index": 0, "is_word_start": True},
     {"generate": "nacotar","core": "cotar", "overlap": "na", "word_index": 0, "is_word_start": False},
     ...
  ]
}
The generator should render seg["generate"]; the stitcher uses seg["overlap"] to align and then
drops the overlap letters from the later image.
"""

MAX_WHOLE = 11      # words up to this length are generated in one piece
OVERLAP = 0      # letters shared between neighbouring chunks (0 disables overlap)
VOWELS = set("aeiouyAEIOUY")
DIGRAPHS = {"ph", "th", "ch", "sh", "ck", "qu", "ng", "gh", "wh", "rh", "ae", "oe", "ei", "ia", "io"}
ONSETS = {"pr", "tr", "gl", "cr", "br", "st", "pl", "gr", "fl", "cl", "dr", "bl", "sc", "sp", "str", "ph", "th", "ch"}

# Latin/taxonomic endings and prefixes that read as a unit; cutting right before/after them is cheap.
SUFFIXES = ("ensis", "oides", "ella", "ata", "atus", "ulus", "inus", "icus", "alis", "ides",
            "us", "is", "um", "ii", "ae", "ia", "ina", "ica")
PREFIXES = ("pseudo", "para", "proto", "micro", "macro", "neo", "pan", "sub", "super", "hyper",
            "cyrt", "platy", "steno", "mega", "poly", "eu", "epi", "hypo", "tele")

# cost of a core chunk by length (lower is better)
_LEN_COST = {5: 0.3, 6: 0.0, 7: 0.0, 8: 0.2}


def _cut_cost(w, i):
    """Cost of cutting between w[i-1] and w[i]."""
    lw = w.lower()
    a, b = lw[i - 1], lw[i]
    if (a + b) in DIGRAPHS:
        return 5.0
    a_v, b_v = a in VOWELS, b in VOWELS
    if a_v and not b_v:       cost = 0.0     # vowel | consonant  (Pi|na)
    elif not a_v and not b_v: cost = 0.6     # consonant | consonant
    elif a_v and b_v:         cost = 1.5     # vowel | vowel
    else:                     cost = 2.0     # consonant | vowel (strands a consonant)
    if any(lw[i:].startswith(s) and len(lw) - i == len(s) for s in SUFFIXES):
        cost -= 0.8                           # cut right before a word-final ending
    if any(lw[:i] == p for p in PREFIXES):
        cost -= 0.8                           # cut right after a known prefix
    return cost


def _chunk_cost(c, first):
    cost = 0.0
    if not any(ch in VOWELS for ch in c):
        cost += 5.0                           # no vowel -> unreadable as a word
    lc = c.lower()
    if not first and len(lc) > 2 and lc[0] not in VOWELS and lc[1] not in VOWELS \
            and lc[:2] not in DIGRAPHS and lc[:2] not in ONSETS:
        cost += 1.5                           # starts with an awkward consonant cluster
    if lc[-1] not in VOWELS and len(lc) > 2 and lc[-2] not in VOWELS and lc[-2:] not in DIGRAPHS:
        cost += 0.4                           # ends mid consonant cluster
    return cost


def _chunk_word(w):
    n = len(w)
    if n <= MAX_WHOLE:
        return [w]
    INF = float("inf")
    best, prev = [INF] * (n + 1), [0] * (n + 1)
    best[0] = 0.0
    for end in range(1, n + 1):
        for length, lc in _LEN_COST.items():
            start = end - length
            if start < 0 or best[start] == INF:
                continue
            cost = best[start] + lc + (_cut_cost(w, start) if start > 0 else 0.0) \
                   + _chunk_cost(w[start:end], start == 0)
            if cost < best[end]:
                best[end], prev[end] = cost, start
    if best[n] == INF:                        # fallback, should not happen
        return [w[i:i + 4] for i in range(0, n, 4)]
    parts, end = [], n
    while end > 0:
        parts.append(w[prev[end]:end])
        end = prev[end]
    return parts[::-1]


def _overlap_for(prev_core, core):
    """Tail of the previous core to prepend. Prefer an overlap that starts on a vowel/consonant
    boundary (a natural join in cursive) and never swallows the whole previous chunk."""
    k = min(OVERLAP, len(prev_core) - 1)
    return prev_core[-k:] if k > 0 else ""


def split_name_into_syllables(name_string):
    segments, cores = [], []
    for wi, word in enumerate(name_string.split()):
        prev = None
        for core in _chunk_word(word):
            ov = _overlap_for(prev, core) if prev else ""
            segments.append({"generate": ov + core, "core": core, "overlap": ov,
                             "word_index": wi, "is_word_start": prev is None})
            cores.append(core)
            prev = core
    return {"original_name": name_string, "digits_or_syllables": cores, "segments": segments}


if __name__ == "__main__":
    for n in ["Pinacotarsus proximus", "Troglophyes bedeli", "Stenelmis grossepunctata", "Aphodius rufus"]:
        d = split_name_into_syllables(n)
        print(n, "->", [(s["overlap"], s["core"]) for s in d["segments"]])