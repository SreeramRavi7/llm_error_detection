"""Phase 1 run: greedy answers, grades, baselines, hidden states.

Pilot (start here):
    python -m scripts.pilot --dataset triviaqa --n 200
    python -m scripts.pilot --dataset gsm8k --n 200

Full run later: same command with a bigger --n and --tag full.
Saves after every batch group, and resumes if you run it again after a crash.

Outputs (in --out-dir):
    <model>_<dataset>_<tag>.jsonl        one Row per line
    <model>_<dataset>_<tag>_hidden.npz   ids, h_question, h_answer  (n, layers+1, hidden), fp16
    <model>_<dataset>_<tag>_check.csv    30 random rows to hand-check the grading
    <model>_<dataset>_<tag>_summary.txt  accuracy, baseline AUROCs, quick probe check
"""

import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from src.data import load
from src.features import chunks, parse_confidence
from src.grading import grade
from src.model_utils import feature_pass, generate_batch, load_model
from src.prompts import MODELS, VERBAL_CONF, SYSTEM, build_messages, max_new_tokens
from src.report import summarize
from src.schema import read_jsonl, write_jsonl


def save(rows, hq, ha, paths):
    write_jsonl(rows, paths["jsonl"])
    np.savez(paths["npz"], ids=np.array([r.id for r in rows]),
             h_question=np.stack(hq), h_answer=np.stack(ha))


def load_progress(paths):
    if not (os.path.exists(paths["jsonl"]) and os.path.exists(paths["npz"])):
        return [], [], []
    done = read_jsonl(paths["jsonl"])
    z = np.load(paths["npz"])
    if not done or list(z["ids"]) != [r.id for r in done]:
        print("Saved files don't match, starting over")
        return [], [], []
    print(f"Resuming: {len(done)} rows already done")
    return done, list(z["h_question"]), list(z["h_answer"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["triviaqa", "gsm8k"], required=True)
    ap.add_argument("--model", default="qwen1.5b", choices=list(MODELS))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--batch", type=int, default=None, help="default 32 for triviaqa, 16 for gsm8k")
    ap.add_argument("--save-every", type=int, default=4, help="save after this many batches")
    ap.add_argument("--no-verbal", action="store_true", help="skip the stated-confidence baseline")
    args = ap.parse_args()
    bs = args.batch or (32 if args.dataset == "triviaqa" else 16)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"{args.model}_{args.dataset}_{args.tag}")
    paths = {k: base + s for k, s in
             [("jsonl", ".jsonl"), ("npz", "_hidden.npz"), ("csv", "_check.csv"), ("txt", "_summary.txt")]}

    rows = load(args.dataset, n=args.n)
    done, hq, ha = load_progress(paths)
    done_ids = {r.id for r in done}
    todo = [r for r in rows if r.id not in done_ids]

    if todo:
        tok, model, stop_ids = load_model(args.model)
        print(f"{len(todo)} to do, batch size {bs}, on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        for b, batch in enumerate(tqdm(list(chunks(todo, bs)), desc=f"{args.model} {args.dataset}")):
            t0 = time.time()
            msgs = [build_messages(r.dataset, r.question) for r in batch]
            p_ids, a_ids, answers = generate_batch(tok, model, stop_ids, msgs, max_new_tokens(args.dataset))

            for r, p, a, ans in zip(batch, p_ids, a_ids, answers):
                mean_lp, h_q, h_a = feature_pass(model, p, a)
                r.model = args.model
                r.greedy_answer = ans
                r.correct = grade(r.dataset, ans, r.gold_answers)
                r.mean_logprob = mean_lp
                r.n_answer_tokens = len(a)
                hq.append(h_q)
                ha.append(h_a)
            per_q = (time.time() - t0) / len(batch)

            if not args.no_verbal:
                vmsgs = [[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": VERBAL_CONF.format(q=r.question, a=r.greedy_answer)}]
                         for r in batch]
                _, _, replies = generate_batch(tok, model, stop_ids, vmsgs, 8)
                for r, reply in zip(batch, replies):
                    r.verbal_conf = parse_confidence(reply)

            for r in batch:
                r.greedy_time_s = round(per_q, 4)
            done.extend(batch)
            if (b + 1) % args.save_every == 0:
                save(done, hq, ha, paths)
        save(done, hq, ha, paths)

    # Regrade with the current grader, so a grading fix only needs a rerun of this
    # command (no new generation, it resumes and skips straight here)
    for r in done:
        r.correct = grade(r.dataset, r.greedy_answer, r.gold_answers)
    write_jsonl(done, paths["jsonl"])

    text = summarize(done, np.stack(hq), np.stack(ha), paths["csv"])
    with open(paths["txt"], "w") as f:
        f.write(text + "\n")
    print(f"\nSaved: {paths['jsonl']}, {paths['npz']}, {paths['csv']}, {paths['txt']}")


if __name__ == "__main__":
    main()
