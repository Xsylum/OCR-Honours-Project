# syllable_splitter.py
"""
Splits a name into word-sized chunks for DiffusionPen.

DiffusionPen was trained on whole IAM words, so it draws chunks of ~3-5 letters
best. Words of MAX_WHOLE letters or fewer are kept whole; longer words are cut
into chunks (preferably 4 letters) at the most natural-looking cut points.
No external dependencies. Same return shape as before.
"""

MAX_WHOLE = 6                 # words up to this length are generated in one piece
IDEAL = 4                     # preferred chunk length
VOWELS = set("aeiouyAEIOUY")
DIGRAPHS = {"ph", "th", "ch", "sh", "ck", "qu", "ng", "gh", "wh"}   # never cut inside these

# cost of a chunk by length (lower is better); lengths not listed are not allowed
_LEN_COST = {2: 3.0, 3: 0.3, 4: 0.0, 5: 0.3, 6: 1.0, 7: 3.0}


def _cut_cost(w, i):
    """Cost of cutting between w[i-1] and w[i]."""
    a, b = w[i - 1].lower(), w[i].lower()
    if (a + b) in DIGRAPHS:
        return 5.0
    a_v, b_v = a in VOWELS, b in VOWELS
    if a_v and not b_v:   return 0.0    # vowel | consonant  (Pi|na, co|tar)
    if not a_v and not b_v: return 0.5  # consonant | consonant
    if a_v and b_v:       return 1.5    # vowel | vowel
    return 2.0                          # consonant | vowel (strands a consonant)


def _chunk_cost(c, first):
    """Penalise chunks with no vowel, or (after the first) starting with a consonant cluster."""
    cost = 0.0
    if not any(ch in VOWELS for ch in c):
        cost += 5.0
    if not first and len(c) > 1 and c[0].lower() not in VOWELS and c[1].lower() not in VOWELS \
            and c[:2].lower() not in DIGRAPHS and c[:2].lower() not in ("pr", "tr", "gl", "cr", "br", "st", "pl", "gr", "fl"):
        cost += 1.5
    return cost


def _chunk_word(w):
    n = len(w)
    if n <= MAX_WHOLE:
        return [w]
    INF = float("inf")
    best = [INF] * (n + 1)
    prev = [0] * (n + 1)
    best[0] = 0.0
    for end in range(1, n + 1):
        for length, lc in _LEN_COST.items():
            start = end - length
            if start < 0 or best[start] == INF:
                continue
            cost = best[start] + lc + (_cut_cost(w, start) if start > 0 else 0.0) + _chunk_cost(w[start:end], start == 0)
            if cost < best[end]:
                best[end], prev[end] = cost, start
    if best[n] == INF:                       # fallback, should not happen
        return [w[i:i + IDEAL] for i in range(0, n, IDEAL)]
    parts, end = [], n
    while end > 0:
        parts.append(w[prev[end]:end])
        end = prev[end]
    return parts[::-1]


def split_name_into_syllables(name_string):
    """
    "Pinacotarsus proximus" -> {"original_name": ..., "digits_or_syllables": [...]}
    Chunks are returned in reading order; word boundaries are recovered later
    by the stitcher from the original name.
    """
    chunks = []
    for word in name_string.split():
        chunks.extend(_chunk_word(word))
    return {"original_name": name_string, "digits_or_syllables": chunks}


if __name__ == "__main__":
    for n in ["Pinacotarsus proximus", "Troglophyes bedeli", "Stigmatium", "Aphodius rufus"]:
        print(n, "->", split_name_into_syllables(n)["digits_or_syllables"])