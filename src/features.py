"""Small helpers with no torch dependency, so they can be unit tested anywhere."""

import re

_NUM = re.compile(r"\d+(?:\.\d+)?")


def parse_confidence(text: str):
    """First number in the model's reply, as 0 to 100. None if missing or out of range."""
    m = _NUM.search(text or "")
    if not m:
        return None
    v = float(m.group())
    if 0 <= v <= 1 and "." in m.group():   # model answered 0.8 instead of 80
        v *= 100
    return v if 0 <= v <= 100 else None


def cut_at_eos(token_ids, stop_ids):
    """Drop everything from the first stop token onward."""
    stop = set(stop_ids)
    for i, t in enumerate(token_ids):
        if t in stop:
            return token_ids[:i]
    return token_ids


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
