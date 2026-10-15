"""Run this first on Colab or ARC. Checks GPU, loads both datasets, and does one
Qwen generation end to end. If this prints ALL GOOD, setup is done.

    python -m scripts.check_env
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load
from src.grading import grade
from src.prompts import MODELS, build_messages, max_new_tokens


def main():
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}, {p.total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: no GPU. On Colab: Runtime > Change runtime type > T4 GPU")

    for name in ("triviaqa", "gsm8k"):
        rows = load(name, n=200)
        n_test = sum(r.split == "test" for r in rows)
        print(f"{name}: loaded {len(rows)} rows ({n_test} test). Example: {rows[0].question[:70]!r}")

    model_id = MODELS["qwen1.5b"]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto")
    print("model layers:", model.config.num_hidden_layers, "| hidden size:", model.config.hidden_size)

    row = load("triviaqa", n=5)[0]
    msgs = build_messages(row.dataset, row.question)
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model.device)
    t0 = time.time()
    out = model.generate(**enc, max_new_tokens=max_new_tokens(row.dataset), do_sample=False)
    ans = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"Q: {row.question}\nA: {ans}\nGold: {row.gold_answers[:3]}  correct={grade(row.dataset, ans, row.gold_answers)}")
    print(f"generation took {time.time() - t0:.1f}s")
    print("ALL GOOD")


if __name__ == "__main__":
    main()
