"""Semantic entropy: group sampled answers by meaning, then measure how spread out they are.

No torch here, so it can be unit tested anywhere. The optional NLI clusterer is in nli.py.

GSM8K: two answers mean the same thing if their final numbers match.
TriviaQA: two answers mean the same thing if either one matches the other under the same
rule the grader uses (normalized text, one contained in the other as whole words).
"""

import math

from .grading import extract_number, normalize


def _same_trivia(a, b):
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return False
    return a == b or f" {a} " in f" {b} " or f" {b} " in f" {a} "


def cluster_answers(dataset, answers):
    """Cluster id per answer. Greedy: each answer joins the first cluster whose first member matches."""
    ids, reps = [], []
    for ans in answers:
        if dataset == "gsm8k":
            key = extract_number(ans)
            match = None if key is None else next(
                (i for i, r in enumerate(reps) if r is not None and abs(r - key) < 1e-6), None)
        else:
            key = ans
            match = next((i for i, r in enumerate(reps) if _same_trivia(r, key)), None)
        if match is None:                      # unparseable GSM8K answers each get their own cluster
            reps.append(key)
            match = len(reps) - 1
        ids.append(match)
    return ids


def discrete_entropy(cluster_ids):
    """Entropy (nats) of the cluster sizes. 0 = all samples agree, log(k) = all different."""
    n = len(cluster_ids)
    if n == 0:
        return None
    counts = {}
    for c in cluster_ids:
        counts[c] = counts.get(c, 0) + 1
    return -sum((m / n) * math.log(m / n) for m in counts.values())
