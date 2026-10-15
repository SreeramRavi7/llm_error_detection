"""Load TriviaQA and GSM8K into Row objects with a fixed train/test split.

The split is decided by a seeded shuffle, so everyone gets the same split
on any machine. Never re-split later, or results won't line up.
"""

import random
from datasets import load_dataset

from .schema import Row

SEED = 0
TEST_FRAC = 0.2


def _assign_splits(rows):
    rng = random.Random(SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_test = int(len(rows) * TEST_FRAC)
    test = set(idx[:n_test])
    for i, r in enumerate(rows):
        r.split = "test" if i in test else "train"
    return rows


def load_triviaqa(n=10000):
    # rc.nocontext = questions only, no evidence documents (much smaller download)
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    rows = [
        Row(
            id=f"tqa_{ex['question_id']}",
            dataset="triviaqa",
            question=ex["question"],
            gold_answers=list(dict.fromkeys([ex["answer"]["value"], *ex["answer"]["aliases"]])),
            split="",
        )
        for ex in ds
    ]
    return _assign_splits(rows)


def load_gsm8k(n=None):
    # Use both official splits as one pool, then apply our own split
    rows = []
    for part in ("train", "test"):
        ds = load_dataset("openai/gsm8k", "main", split=part)
        for i, ex in enumerate(ds):
            gold = ex["answer"].split("####")[-1].strip()
            rows.append(
                Row(id=f"gsm_{part}_{i}", dataset="gsm8k",
                    question=ex["question"], gold_answers=[gold], split="")
            )
    if n:
        random.Random(SEED).shuffle(rows)
        rows = rows[:n]
    return _assign_splits(rows)


def load(name, n=None):
    if name == "triviaqa":
        return load_triviaqa(n or 10000)
    if name == "gsm8k":
        return load_gsm8k(n)
    raise ValueError(name)
