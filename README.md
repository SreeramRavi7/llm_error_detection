# Knowing When the Model Is Wrong

ECE 5424 term project. We compare two ways to catch a small LLM's wrong answers:
a probe on hidden states (one run) vs. semantic entropy (many runs).

## Setup

**Colab (start here)**
1. Runtime > Change runtime type > **T4 GPU**
2. Open `notebooks/setup_colab.ipynb` and run all cells.
3. It should end with `ALL GOOD`.

**Local or ARC**
```bash
git clone <repo-url> && cd llm-error-detect
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests          # grading tests, no GPU needed
python -m scripts.check_env        # needs a GPU
```

**Llama access:** Llama-3.2-3B is gated. Each person: accept the license at
huggingface.co/meta-llama/Llama-3.2-3B-Instruct, then `huggingface-cli login`.
Do this now, approval is not always instant.

## Layout

```
src/
  schema.py    the shared row format (read this first)
  data.py      loads TriviaQA + GSM8K with a fixed train/test split
  prompts.py   all prompts and model names, in one place
  grading.py   right/wrong checks for both datasets
scripts/       runnable steps (check_env now, more per phase)
tests/         run before every push
outputs/       generated .jsonl and .npy files (not committed, share via Drive)
```

## Rules so we don't break each other's work

- Everyone uses the `Row` format in `src/schema.py`. Add a field there if you need one, don't make side files.
- Never change the split (`SEED` in `data.py`) once we start generating.
- Prompts only change in `src/prompts.py`, and tell the group if you do.
- Hidden states: `outputs/<model>_<dataset>_hidden.npy`, row i matches line i of the `.jsonl`.
- Big files go in the shared Drive folder, not git.

## Who does what

| Phase | Task | Owner |
|---|---|---|
| 1 | Generate greedy answers, grade, baselines, save hidden states | Sreeram |
| 2 | Probes per layer, layer study | Trish |
| 2 | Sampling + semantic entropy | Name 2 |
| 3 | Head-to-head, cost study, transfer | All |

## Next step

Pilot on 200 TriviaQA + 200 GSM8K questions before running everything.
We need a decent mix of right and wrong answers, and we need to hand-check
about 30 grades per dataset to make sure grading isn't broken.
