"""Shared data format. Everyone reads and writes rows in this shape.

One .jsonl file per (model, dataset), e.g. outputs/qwen1.5b_triviaqa.jsonl
Hidden states go in a matching .npz file, row i of the .npz = line i of the .jsonl.

Fields get filled in stages. A field is None until its stage has run.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class Row:
    # Stage 0: from the dataset loader
    id: str
    dataset: str                      # "triviaqa" or "gsm8k"
    question: str
    gold_answers: list                # all accepted answers (aliases for TriviaQA, one number for GSM8K)
    split: str                        # "train" or "test"

    # Stage 1: greedy generation + grading (Sreeram)
    model: Optional[str] = None
    greedy_answer: Optional[str] = None
    correct: Optional[int] = None     # 1 = right, 0 = wrong. Label for everything else is wrong = 1 - correct
    n_answer_tokens: Optional[int] = None
    greedy_time_s: Optional[float] = None  # GPU seconds for the greedy pass + feature pass, per question

    # Stage 1b: baselines (Sreeram)
    mean_logprob: Optional[float] = None
    verbal_conf: Optional[float] = None   # 0 to 100, None if the model gave no number

    # Stage 2: sampling (Name 2)
    samples: list = field(default_factory=list)
    semantic_entropy: Optional[float] = None

    # Stage 2: probe score, filled at eval time (Trish)
    probe_score: Optional[float] = None


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r) if isinstance(r, Row) else r) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [Row(**json.loads(line)) for line in f if line.strip()]
