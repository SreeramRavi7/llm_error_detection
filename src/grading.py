"""Grading: decide if a model answer is right.

TriviaQA: normalize text (lowercase, drop articles/punctuation) and check
whether any gold alias appears in the answer, or the answer equals an alias.
GSM8K: find the final answer and compare numbers. Order of preference:
\boxed{...}, then text after "####", then a **bold** number on the last line,
then the last number anywhere. The model often ignores the format we ask for,
and just taking the last number fails on endings like "480 cans in 8 hours".
"""

import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def grade_triviaqa(answer: str, gold_answers: list) -> int:
    pred = normalize(answer)
    if not pred:
        return 0
    padded = f" {pred} "
    for g in gold_answers:
        g = normalize(g)
        if g and (pred == g or f" {g} " in padded):
            return 1
    return 0


def extract_number(text: str):
    """Final answer as a float. GSM8K gold uses '#### 42'."""
    boxed = [b for b in _BOXED.findall(text) if _NUM.search(b)]
    last_line = text.strip().split("\n")[-1]          # skip bold step headers like **Step 3:**
    bold = [b for b in _BOLD.findall(last_line) if _NUM.search(b)]
    if boxed:
        text = boxed[-1]
    elif "####" in text:
        text = text.split("####")[-1]
    elif bold:
        text = bold[-1]
    nums = _NUM.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def grade_gsm8k(answer: str, gold_answers: list) -> int:
    pred = extract_number(answer)
    gold = extract_number(str(gold_answers[0]))
    if pred is None or gold is None:
        return 0
    return int(abs(pred - gold) < 1e-6)


def grade(dataset: str, answer: str, gold_answers: list) -> int:
    if dataset == "triviaqa":
        return grade_triviaqa(answer, gold_answers)
    if dataset == "gsm8k":
        return grade_gsm8k(answer, gold_answers)
    raise ValueError(f"unknown dataset {dataset}")
