"""Phase 3: put all 4 methods side by side, check the tradeoff rule, draw the figures.

No GPU needed. Reads, from --in-dir (searched recursively, so a Kaggle input folder works):
    <model>_<dataset>_full_se.jsonl                       semantic entropy run (temperature 1.0)
    <model>_<dataset>_full_probe_scores.jsonl             probe score per test row
    <model>_<dataset>_full_probe_layers.csv               layer study (optional)
    <model>_gsm8k_from_triviaqa_full_probe_scores.jsonl   transfer test (optional)

    python -m scripts.compare --in-dir outputs --out-dir results

Writes to --out-dir:
    main_table.csv / main_table.md   AUROC + 95% CI for each method, cost, tradeoff verdict
    transfer.md                      probe trained on trivia, tested on math
    cost_vs_auroc.png                the main figure
    layers.png                       probe test AUROC at every layer
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

MODELS = ["qwen1.5b", "llama3b"]
DATASETS = ["triviaqa", "gsm8k"]
NAMES = {"qwen1.5b": "Qwen2.5-1.5B", "llama3b": "Llama-3.2-3B", "triviaqa": "TriviaQA", "gsm8k": "GSM8K"}
METHODS = ["probe", "semantic_entropy", "log_prob", "stated_conf"]
LABELS = {"probe": "Probe", "semantic_entropy": "Semantic entropy",
          "log_prob": "Log-prob", "stated_conf": "Stated confidence"}
COLORS = {"probe": "#2a78d6", "semantic_entropy": "#eb6834", "log_prob": "#1baf7a", "stated_conf": "#eda100"}
MARGIN = 0.03   # tradeoff rule from the proposal feedback: probe within 0.03 AUROC of semantic entropy


def find(in_dir, name):
    hits = glob.glob(os.path.join(in_dir, "**", name), recursive=True)
    return hits[0] if hits else None


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def boot(y, scores, n=1000, seed=0):
    """95% bootstrap CIs. scores: dict name -> array (NaN = missing). Also CI of probe - SE."""
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in scores}
    diff = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        a = {}
        for k, s in scores.items():
            ok = ~np.isnan(s[i])
            if len(set(y[i][ok])) == 2:
                a[k] = roc_auc_score(y[i][ok], s[i][ok])
                vals[k].append(a[k])
        if "probe" in a and "semantic_entropy" in a:
            diff.append(a["probe"] - a["semantic_entropy"])
    ci = {k: tuple(np.percentile(v, [2.5, 97.5])) if v else (np.nan, np.nan) for k, v in vals.items()}
    return ci, (tuple(np.percentile(diff, [2.5, 97.5])) if diff else (np.nan, np.nan))


def auc(y, s):
    ok = ~np.isnan(s)
    return roc_auc_score(y[ok], s[ok]) if len(set(y[ok])) == 2 else np.nan


def load_pair(in_dir, model, ds, probe_file=None):
    se_path = find(in_dir, f"{model}_{ds}_full_se.jsonl")
    pr_path = find(in_dir, probe_file or f"{model}_{ds}_full_probe_scores.jsonl")
    if not se_path:
        return None
    rows = read_jsonl(se_path)
    probe = {}
    if pr_path:
        probe = {r["id"]: r["probe_score"] for r in read_jsonl(pr_path)}
        missing = sum(r["id"] not in probe for r in rows)
        assert missing == 0, f"{pr_path}: {missing} test rows have no probe score"
    nan = float("nan")
    y = np.array([1 - r["correct"] for r in rows])
    s = {
        "probe": np.array([probe.get(r["id"], nan) for r in rows], dtype=float),
        "semantic_entropy": np.array([r["semantic_entropy"] for r in rows], dtype=float),
        "log_prob": np.array([nan if r["mean_logprob"] is None else -r["mean_logprob"] for r in rows]),
        "stated_conf": np.array([nan if r["verbal_conf"] is None else -r["verbal_conf"] for r in rows]),
    }
    t_greedy = float(np.mean([r["greedy_time_s"] or 0 for r in rows]))
    t_se = float(np.mean([r["se_time_s"] for r in rows]))
    k = len(rows[0]["samples"])
    # GPU seconds per question to get each score. The probe and log-prob both come out of the
    # normal answer pass (greedy + one feature pass). Semantic entropy needs that answer too,
    # plus k samples. Stated confidence is one extra short reply that was not timed separately.
    cost = {"probe": t_greedy, "log_prob": t_greedy, "semantic_entropy": t_greedy + t_se, "stated_conf": np.nan}
    return {"y": y, "s": s, "cost": cost, "k": k, "has_probe": bool(pr_path)}


def fmt(a, ci):
    return "n/a" if np.isnan(a) else f"{a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="outputs")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    csv_rows = []
    md = ["| Model | Dataset | Test Qs (wrong) | Probe | Semantic entropy | Log-prob | Stated conf | "
          "Probe − SE [95% CI] | Probe cost / SE cost | Within 0.03? |",
          "|" + "---|" * 10]
    for m in MODELS:
        for d in DATASETS:
            p = load_pair(args.in_dir, m, d)
            if p is None:
                print(f"skip {m} {d}: no _se.jsonl found")
                continue
            if not p["has_probe"]:
                print(f"note {m} {d}: no probe scores yet, probe column will be n/a")
            y, s = p["y"], p["s"]
            a = {k: auc(y, v) for k, v in s.items()}
            ci, dci = boot(y, s)
            ratio = p["cost"]["probe"] / p["cost"]["semantic_entropy"]
            gap = a["probe"] - a["semantic_entropy"]
            if np.isnan(gap):
                verdict = "n/a"
            elif dci[0] >= -MARGIN:
                verdict = "yes (whole CI)"
            elif gap >= -MARGIN:
                verdict = "yes (point only)"
            else:
                verdict = "no"
            results[(m, d)] = {**p, "auc": a, "ci": ci}
            md.append(f"| {NAMES[m]} | {NAMES[d]} | {len(y)} ({int(y.sum())}) | "
                      + " | ".join(fmt(a[k], ci[k]) for k in METHODS)
                      + (f" | {gap:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]" if not np.isnan(gap) else " | n/a")
                      + f" | 1/{1 / ratio:.1f} | {verdict} |")
            row = {"model": m, "dataset": d, "n_test": len(y), "n_wrong": int(y.sum()), "k": p["k"]}
            for k in METHODS:
                row[f"{k}_auroc"] = round(a[k], 4)
                row[f"{k}_ci_lo"] = round(ci[k][0], 4)
                row[f"{k}_ci_hi"] = round(ci[k][1], 4)
            row.update({"probe_minus_se": round(gap, 4), "diff_ci_lo": round(dci[0], 4),
                        "diff_ci_hi": round(dci[1], 4),
                        "gpu_s_probe": round(p["cost"]["probe"], 3),
                        "gpu_s_se": round(p["cost"]["semantic_entropy"], 3),
                        "cost_ratio": round(ratio, 4), "within_margin": verdict})
            csv_rows.append(row)

    if not csv_rows:
        raise SystemExit("No _se.jsonl files found under " + args.in_dir)
    with open(os.path.join(args.out_dir, "main_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    md += ["", "AUROC for flagging wrong answers, 95% bootstrap CI in brackets (1,000 resamples).",
           f"Within 0.03: 'whole CI' = even the low end of the probe − SE interval is above −{MARGIN}; "
           "'point only' = the estimate is, but the interval reaches lower.",
           "Cost = measured GPU seconds per question on a Kaggle T4. Stated confidence needs one extra "
           "short reply that was not timed."]
    with open(os.path.join(args.out_dir, "main_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    # transfer: probe trained on TriviaQA, tested on GSM8K
    tr = ["| Model | Probe trained on GSM8K | Probe trained on TriviaQA | Drop | Log-prob |", "|---|---|---|---|---|"]
    for m in MODELS:
        p = load_pair(args.in_dir, m, "gsm8k", f"{m}_gsm8k_from_triviaqa_full_probe_scores.jsonl")
        if p is None or not p["has_probe"] or (m, "gsm8k") not in results:
            continue
        a_in = results[(m, "gsm8k")]["auc"]["probe"]
        a_tr = auc(p["y"], p["s"]["probe"])
        tr.append(f"| {NAMES[m]} | {a_in:.3f} | {a_tr:.3f} | {a_tr - a_in:+.3f} | {auc(p['y'], p['s']['log_prob']):.3f} |")
    if len(tr) > 2:
        with open(os.path.join(args.out_dir, "transfer.md"), "w") as f:
            f.write("\n".join(tr) + "\n")
        print("\n" + "\n".join(tr))

    plot_cost(results, os.path.join(args.out_dir, "cost_vs_auroc.png"))
    plot_layers(args.in_dir, results, os.path.join(args.out_dir, "layers.png"))
    print(f"\nWrote results to {args.out_dir}/")


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b8b7b2")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.grid(True, color="#ecebe8", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_cost(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=True)
    shown = ["probe", "log_prob", "semantic_entropy"]
    for i, m in enumerate(MODELS):
        for j, d in enumerate(DATASETS):
            ax = axes[i, j]
            style(ax)
            ax.set_xscale("log")
            ax.set_title(f"{NAMES[m]} · {NAMES[d]}", fontsize=10, color="#0b0b0b", loc="left")
            r = results.get((m, d))
            if r is None:
                continue
            for k in shown:
                a, c = r["auc"][k], r["cost"][k]
                if np.isnan(a):
                    continue
                lo, hi = r["ci"][k]
                ax.errorbar(c, a, yerr=[[a - lo], [hi - a]], fmt="o", ms=8, color=COLORS[k],
                            ecolor=COLORS[k], elinewidth=2, capsize=0,
                            mec="#fcfcfb", mew=2, zorder=3)
                # probe and log-prob share a cost, so label them left and right of the point
                # semantic entropy goes above its error bar so it never runs into the other labels
                off, ha = {"probe": ((-10, 0), "right"), "log_prob": ((10, 0), "left"),
                           "semantic_entropy": ((0, 0), "center")}[k]
                if k == "semantic_entropy":
                    ax.annotate(f"{LABELS[k]} {a:.2f}", (c, hi), xytext=(0, 6), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8.5, color="#52514e")
                    continue
                ax.annotate(f"{LABELS[k]} {a:.2f}", (c, a), xytext=off, textcoords="offset points",
                            ha=ha, va="center", fontsize=8.5, color="#52514e")
            xs = [r["cost"][k] for k in shown]
            ax.set_xlim(min(xs) / 3, max(xs) * 3)
            if j == 0:
                ax.set_ylabel("AUROC (flagging wrong answers)", fontsize=9, color="#52514e")
            if i == 1:
                ax.set_xlabel("GPU seconds per question (log scale)", fontsize=9, color="#52514e")
    # one shared y range from all CI ends, with room on top for the labels above the SE points
    ends = [v for r in results.values() for k in shown if not np.isnan(r["auc"][k]) for v in r["ci"][k]]
    lo_y, hi_y = min(ends), max(ends)
    axes[0, 0].set_ylim(lo_y - 0.02, hi_y + 0.05)
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=COLORS[k], label=LABELS[k]) for k in shown]
    fig.suptitle("Cost vs. error-detection AUROC (95% CI bars)", y=0.985, fontsize=11, color="#0b0b0b")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def plot_layers(in_dir, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    drew = False
    for j, d in enumerate(DATASETS):
        ax = axes[j]
        style(ax)
        ax.set_title(NAMES[d], fontsize=10, color="#0b0b0b", loc="left")
        for m, color in zip(MODELS, ["#2a78d6", "#eb6834"]):
            f = find(in_dir, f"{m}_{d}_full_probe_layers.csv")
            if not f:
                continue
            with open(f) as fh:
                t = [r for r in csv.DictReader(fh) if r["token"] == "answer"]
            L = [int(r["layer"]) for r in t]
            v = [float(r["test_auroc"]) for r in t]
            b = int(np.argmax([float(r["cv_auroc"]) for r in t]))   # same rule as probe.py: pick on train CV
            ax.plot(L, v, color=color, lw=2, label=f"{NAMES[m]} (picked L{L[b]}, test {v[b]:.2f})")
            ax.plot(L[b], v[b], "o", ms=8, color=color, mec="#fcfcfb", mew=2)
            r = results.get((m, d))
            if r is not None:
                ax.axhline(r["auc"]["log_prob"], color=color, lw=1, ls=":", alpha=0.8)
            drew = True
        ax.set_xlabel("Layer (0 = embeddings)", fontsize=9, color="#52514e")
        if j == 0:
            ax.set_ylabel("Probe test AUROC", fontsize=9, color="#52514e")
    if not drew:
        plt.close(fig)
        return
    for ax in axes:
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Probe test AUROC by layer (dot = layer picked on train CV, dotted line = log-prob)", y=0.99, fontsize=11, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


if __name__ == "__main__":
    main()
