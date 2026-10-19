import numpy as np

from src.features import chunks, cut_at_eos, parse_confidence
from src.report import auroc, quick_probe, summarize
from src.schema import Row, read_jsonl, write_jsonl


def test_parse_confidence():
    assert parse_confidence("85") == 85
    assert parse_confidence("I'd say 70%.") == 70
    assert parse_confidence("0.9") == 90
    assert parse_confidence("150") is None
    assert parse_confidence("not sure") is None
    assert parse_confidence(None) is None


def test_cut_and_chunks():
    assert cut_at_eos([5, 6, 2, 7], {2}) == [5, 6]
    assert cut_at_eos([5, 6], {2}) == [5, 6]
    assert [len(c) for c in chunks(list(range(10)), 4)] == [4, 4, 2]


def test_auroc_skips_missing():
    assert auroc([0.1, 0.9, None, 0.8], [0, 1, 1, 1]) == 1.0
    assert np.isnan(auroc([0.1, 0.2], [1, 1]))


def test_quick_probe_finds_planted_signal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    h = rng.normal(size=(200, 5, 16)).astype(np.float16)
    h[:, 3, 0] += 3 * y          # signal only at layer 3
    res = quick_probe(h, y, [0, 3])
    assert res[3] > 0.9 and res[0] < 0.7


def _fake_rows(n=100):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        c = int(rng.random() < 0.5)
        rows.append(Row(id=f"x{i}", dataset="triviaqa", question=f"q{i}", gold_answers=["a"],
                        split="train", model="qwen1.5b", greedy_answer="a" if c else "b", correct=c,
                        mean_logprob=float(-0.2 - (1 - c) + rng.normal(0, 0.5)),
                        verbal_conf=None if i % 10 == 0 else float(rng.integers(50, 100)),
                        greedy_time_s=0.3))
    return rows


def test_summarize_and_roundtrip(tmp_path):
    rows = _fake_rows()
    p = tmp_path / "r.jsonl"
    write_jsonl(rows, p)
    rows = read_jsonl(p)
    h = np.random.default_rng(2).normal(size=(len(rows), 29, 32)).astype(np.float16)
    text = summarize(rows, h, h, str(tmp_path / "check.csv"))
    assert "accuracy" in text and "Quick probe" in text
    assert (tmp_path / "check.csv").read_text().count("\n") == 31
