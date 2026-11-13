"""Phase 2: semantic entropy on the test split.

Reads the Phase 1 file (<model>_<dataset>_<tag>.jsonl), drops repeated ids, keeps the test
split, samples k answers per question, clusters them by meaning, and scores the entropy.

    python -m scripts.sample --dataset triviaqa --model qwen1.5b
    python -m scripts.sample --dataset gsm8k --model llama3b

Writes <model>_<dataset>_<tag>_se.jsonl (_se_t0.7.jsonl etc. if --temperature is not 1.0) (test rows with samples, semantic_entropy, n_clusters,
se_time_s) and _se_summary.txt. Saves as it goes and resumes after a crash.
"""

import argparse
import os
import time

import torch
from tqdm import tqdm

from src.features import chunks
from src.model_utils import load_model, sample_batch
from src.prompts import build_messages, max_new_tokens
from src.grading import grade
from src.report import auroc
from src.schema import keep_mask, read_jsonl, write_jsonl
from src.semantic import cluster_answers, discrete_entropy


def sample_acc(rows):
    n = sum(len(r.samples) for r in rows)
    return sum(grade(r.dataset, a, r.gold_answers) for r in rows for a in r.samples) / max(1, n)


def summarize(rows, k, temperature):
    wrong = [1 - r.correct for r in rows]
    se = [r.semantic_entropy for r in rows]
    lp = [None if r.mean_logprob is None else -r.mean_logprob for r in rows]
    vc = [None if r.verbal_conf is None else -r.verbal_conf for r in rows]
    t_se = sum(r.se_time_s for r in rows) / len(rows)
    t_g = sum(r.greedy_time_s or 0 for r in rows) / len(rows)
    lines = [
        f"{len(rows)} test rows | accuracy {1 - sum(wrong) / len(rows):.1%} | wrong {sum(wrong)} | k={k} | temperature {temperature}",
        f"AUROC semantic entropy:       {auroc(se, wrong):.3f}",
        f"AUROC token log-prob:         {auroc(lp, wrong):.3f}   (same rows)",
        f"AUROC stated confidence:      {auroc(vc, wrong):.3f}   (same rows)",
        f"Sample accuracy:              {sample_acc(rows):.1%}   (greedy {1 - sum(wrong) / len(rows):.1%}; far below greedy = samples too random, lower --temperature)",
        f"Mean clusters per question:   {sum(r.n_clusters for r in rows) / len(rows):.2f}",
        f"All {k} samples agree:          {sum(r.n_clusters == 1 for r in rows) / len(rows):.1%} of questions",
        f"GPU s/question: greedy+features {t_g:.2f} | sampling {t_se:.2f}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["triviaqa", "gsm8k"], required=True)
    ap.add_argument("--model", default="qwen1.5b")
    ap.add_argument("--tag", default="full", help="which Phase 1 run to read")
    ap.add_argument("--in-dir", default="outputs", help="folder with the Phase 1 .jsonl")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N test questions (for a quick test)")
    ap.add_argument("--batch", type=int, default=None, help="questions per batch, default 16 trivia / 4 gsm8k")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--nli", action="store_true", help="TriviaQA only: cluster with DeBERTa NLI instead of string match")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    bs = args.batch or (16 if args.dataset == "triviaqa" else 4)

    base = f"{args.model}_{args.dataset}_{args.tag}"
    rows = read_jsonl(os.path.join(args.in_dir, base + ".jsonl"))
    rows = [r for r, k in zip(rows, keep_mask(rows)) if k and r.split == "test"]
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    se_base = base + "_se" + ("" if args.temperature == 1.0 else f"_t{args.temperature:g}")
    out_path = os.path.join(args.out_dir, se_base + ".jsonl")
    done = read_jsonl(out_path) if os.path.exists(out_path) else []
    done_ids = {r.id for r in done}
    todo = [r for r in rows if r.id not in done_ids]
    print(f"{len(rows)} test questions, {len(done)} already done, {len(todo)} to do")

    if todo:
        torch.manual_seed(args.seed)
        tok, model, stop_ids = load_model(args.model)
        nli = None
        if args.nli and args.dataset == "triviaqa":
            from src.nli import NLIClusterer
            nli = NLIClusterer()
        limit = max_new_tokens(args.dataset)
        hit_total = 0
        for b, batch in enumerate(tqdm(list(chunks(todo, bs)), desc=f"{args.model} {args.dataset} k={args.k}")):
            t0 = time.time()
            msgs = [build_messages(r.dataset, r.question) for r in batch]
            results = sample_batch(tok, model, stop_ids, msgs, limit, args.k, args.temperature)
            per_q = (time.time() - t0) / len(batch)
            for r, (answers, hit) in zip(batch, results):
                ids = nli.cluster(r.question, answers) if nli else cluster_answers(r.dataset, answers)
                r.samples = answers
                r.n_clusters = len(set(ids))
                r.semantic_entropy = discrete_entropy(ids)
                r.se_time_s = round(per_q, 4)
                hit_total += hit
            done.extend(batch)
            if (b + 1) % args.save_every == 0:
                write_jsonl(done, out_path)
        write_jsonl(done, out_path)
        if hit_total:
            print(f"NOTE: {hit_total} samples hit the {limit}-token limit")

    text = summarize(done, args.k, args.temperature)
    print("\n" + text)
    with open(os.path.join(args.out_dir, se_base + "_summary.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
