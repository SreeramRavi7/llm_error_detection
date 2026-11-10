"""Pilot summary: accuracy, baseline AUROCs, a quick probe check, and a hand-check CSV.

No torch here, so it runs anywhere (laptop, Kaggle CPU) on saved outputs.
"""

import csv
import math
import random

import numpy as np


def auroc(scores, labels):
    """AUROC for predicting WRONG answers. Higher score should mean more likely wrong."""
    from sklearn.metrics import roc_auc_score
    pairs = [(s, y) for s, y in zip(scores, labels)
             if s is not None and not (isinstance(s, float) and math.isnan(s))]
    ys = [y for _, y in pairs]
    if len(set(ys)) < 2:
        return float("nan")
    return roc_auc_score(ys, [s for s, _ in pairs])


def quick_probe(h, wrong, layers, folds=5, seed=0):
    """5-fold CV AUROC of a logistic regression probe at the given layers.

    Just a sanity check on pilot data: if this is near 0.5 at every layer, something is off.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(wrong)
    if len(set(y)) < 2 or min(np.bincount(y)) < folds:
        return {}
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    out = {}
    for layer in layers:
        X = h[:, layer].astype(np.float32)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
        p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        out[layer] = auroc(list(p), list(y))
    return out


def summarize(rows, h_question=None, h_answer=None, csv_path=None):
    from .schema import keep_mask
    keep = np.array(keep_mask(rows), dtype=bool)
    n_dup = int((~keep).sum())
    if n_dup:
        rows = [r for r, k in zip(rows, keep) if k]
        h_question = None if h_question is None else h_question[keep]
        h_answer = None if h_answer is None else h_answer[keep]

    wrong = [1 - r.correct for r in rows]
    acc = sum(r.correct for r in rows) / len(rows)
    n_conf = sum(r.verbal_conf is not None for r in rows)
    lines = [f"{len(rows)} rows | accuracy {acc:.1%} | wrong {sum(wrong)}"]
    if n_dup:
        lines.append(f"Dropped {n_dup} repeated question ids (kept the first copy)")

    lp = [None if r.mean_logprob is None else -r.mean_logprob for r in rows]
    vc = [None if r.verbal_conf is None else -r.verbal_conf for r in rows]
    lines.append(f"AUROC from token log-prob:    {auroc(lp, wrong):.3f}")
    lines.append(f"AUROC from stated confidence: {auroc(vc, wrong):.3f}   ({n_conf}/{len(rows)} gave a number)")

    times = [r.greedy_time_s for r in rows if r.greedy_time_s]
    if times:
        lines.append(f"GPU time per question: {np.mean(times):.2f}s")

    if not 0.15 <= acc <= 0.85:
        lines.append("NOTE: accuracy is lopsided. Probes need both classes, check prompts or grading.")

    for name, h in (("answer token", h_answer), ("question token", h_question)):
        if h is not None and len(h) == len(rows):
            n_layers = h.shape[1]
            layers = sorted(set(list(range(0, n_layers, 4)) + [n_layers - 1]))
            res = quick_probe(h, wrong, layers)
            if res:
                lines.append(f"Quick probe AUROC ({name}), 5-fold CV: " +
                             "  ".join(f"L{l}={v:.2f}" for l, v in res.items()))

    if csv_path:
        sample = random.Random(0).sample(rows, min(30, len(rows)))
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "question", "gold", "model_answer", "graded_correct", "your_check"])
            for r in sample:
                w.writerow([r.id, r.question, " | ".join(r.gold_answers[:5]),
                            r.greedy_answer, r.correct, ""])
        lines.append(f"Hand-check file: {csv_path}")

    text = "\n".join(lines)
    print("\n" + text)
    return text
