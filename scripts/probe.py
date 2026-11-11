"""Phase 2: hidden-state probes, done properly. No GPU needed (numpy + scikit-learn).

For every layer: train a logistic regression on the train split, score it on the test split.
The best layer is picked by 5-fold CV on train only, so the test number is not cherry-picked.
Repeated ids are dropped first (see keep_mask).

    python -m scripts.probe --dataset triviaqa --model qwen1.5b
    python -m scripts.probe --dataset gsm8k --model llama3b --train-dataset triviaqa   # transfer

Writes (in --out-dir):
    <base>_probe_layers.csv     layer, token, cv_auroc (train), test_auroc
    <base>_probe_summary.txt    best layer + test AUROC with 95% bootstrap CIs, vs baselines
    <base>_probe_scores.jsonl   {"id", "probe_score"} for every test row, best layer, answer token
"""

import argparse
import csv
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.schema import keep_mask, read_jsonl


def load(in_dir, model, dataset, tag):
    base = os.path.join(in_dir, f"{model}_{dataset}_{tag}")
    rows = read_jsonl(base + ".jsonl")
    z = np.load(base + "_hidden.npz")
    assert list(z["ids"]) == [r.id for r in rows], "hidden states don't line up with the .jsonl"
    keep = np.array(keep_mask(rows), dtype=bool)
    rows = [r for r, k in zip(rows, keep) if k]
    return rows, {"answer": z["h_answer"][keep], "question": z["h_question"][keep]}


def make_probe(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def boot_ci(y, scores, n=1000, seed=0):
    """95% bootstrap CI of AUROC. scores: list of arrays, CI for each and for scores[0] - scores[1]."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    vals = [[] for _ in scores]
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        a = [roc_auc_score(y[i], s[i]) for s in scores]
        for v, x in zip(vals, a):
            v.append(x)
        if len(a) > 1:
            diffs.append(a[0] - a[1])
    ci = [tuple(np.percentile(v, [2.5, 97.5])) for v in vals]
    return ci, (tuple(np.percentile(diffs, [2.5, 97.5])) if diffs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["triviaqa", "gsm8k"], required=True, help="test on this")
    ap.add_argument("--train-dataset", choices=["triviaqa", "gsm8k"], default=None,
                    help="train on this one instead (transfer test). Default: same as --dataset")
    ap.add_argument("--model", default="qwen1.5b")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--in-dir", default="outputs")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--C", type=float, default=0.05, help="inverse L2 strength, smaller = stronger")
    ap.add_argument("--layers", default="all", help='"all" or e.g. "0,8,16,24,28"')
    args = ap.parse_args()
    train_ds = args.train_dataset or args.dataset

    te_rows, te_h = load(args.in_dir, args.model, args.dataset, args.tag)
    if train_ds == args.dataset:
        tr_rows, tr_h = te_rows, te_h
    else:
        tr_rows, tr_h = load(args.in_dir, args.model, train_ds, args.tag)
    tr = np.array([r.split == "train" for r in tr_rows])
    te = np.array([r.split == "test" for r in te_rows])
    y_tr = np.array([1 - r.correct for r in tr_rows])[tr]
    y_te = np.array([1 - r.correct for r in te_rows])[te]
    n_layers = te_h["answer"].shape[1]
    layers = range(n_layers) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    print(f"train: {train_ds} {tr.sum()} rows ({y_tr.sum()} wrong) | test: {args.dataset} {te.sum()} rows ({y_te.sum()} wrong)")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    table, test_scores = [], {}
    for token in ("answer", "question"):
        for L in layers:
            X_tr = tr_h[token][tr, L].astype(np.float32)
            X_te = te_h[token][te, L].astype(np.float32)
            p_cv = cross_val_predict(make_probe(args.C), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
            clf = make_probe(args.C).fit(X_tr, y_tr)
            p_te = clf.predict_proba(X_te)[:, 1]
            cv_auc, te_auc = roc_auc_score(y_tr, p_cv), roc_auc_score(y_te, p_te)
            table.append({"layer": L, "token": token, "cv_auroc": round(cv_auc, 4), "test_auroc": round(te_auc, 4)})
            test_scores[(token, L)] = p_te
            print(f"  {token:8s} L{L:<2d} cv {cv_auc:.3f}  test {te_auc:.3f}")

    tag = args.dataset if train_ds == args.dataset else f"{args.dataset}_from_{train_ds}"
    base = os.path.join(args.out_dir, f"{args.model}_{tag}_{args.tag}")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(base + "_probe_layers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "token", "cv_auroc", "test_auroc"])
        w.writeheader()
        w.writerows(table)

    # pick the best answer-token layer on TRAIN cv, report it on test
    best = max((t for t in table if t["token"] == "answer"), key=lambda t: t["cv_auroc"])
    p_best = test_scores[("answer", best["layer"])]
    test_rows = [r for r, k in zip(te_rows, te) if k]
    lp = np.array([-(r.mean_logprob if r.mean_logprob is not None else 0.0) for r in test_rows])
    lines = [f"{args.model} | train {train_ds} -> test {args.dataset} | {te.sum()} test rows, {y_te.sum()} wrong | C={args.C}",
             f"Best layer (chosen on train CV): L{best['layer']} answer token"]
    (ci_p, ci_lp), ci_d = boot_ci(y_te, [p_best, lp])
    lines.append(f"Probe test AUROC:     {best['test_auroc']:.3f}  95% CI [{ci_p[0]:.3f}, {ci_p[1]:.3f}]")
    lines.append(f"Log-prob test AUROC:  {roc_auc_score(y_te, lp):.3f}  95% CI [{ci_lp[0]:.3f}, {ci_lp[1]:.3f}]")
    lines.append(f"Probe minus log-prob: {best['test_auroc'] - roc_auc_score(y_te, lp):+.3f}  95% CI [{ci_d[0]:+.3f}, {ci_d[1]:+.3f}]")
    vc_rows = [(i, r.verbal_conf) for i, r in enumerate(test_rows) if r.verbal_conf is not None]
    if vc_rows:
        idx = [i for i, _ in vc_rows]
        lines.append(f"Stated conf AUROC:    {roc_auc_score(y_te[idx], [-v for _, v in vc_rows]):.3f}  ({len(idx)} rows with a number)")
    text = "\n".join(lines)
    print("\n" + text)
    with open(base + "_probe_summary.txt", "w") as f:
        f.write(text + "\n")
    with open(base + "_probe_scores.jsonl", "w") as f:
        for r, s in zip(test_rows, p_best):
            f.write(json.dumps({"id": r.id, "probe_score": float(s)}) + "\n")


if __name__ == "__main__":
    main()
