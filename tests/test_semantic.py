import math

from src.semantic import cluster_answers, discrete_entropy


def test_gsm8k_clusters_by_final_number():
    ans = ["... \\boxed{12}", "so the answer is 12.", "#### 12", "\\boxed{13}", "no idea"]
    ids = cluster_answers("gsm8k", ans)
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] != ids[0]
    assert ids[4] not in ids[:4]                      # unparseable gets its own cluster


def test_trivia_clusters_by_matching():
    ids = cluster_answers("triviaqa", ["Paris", "paris.", "Paris, France", "London", "The London"])
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] == ids[4] != ids[0]
    assert cluster_answers("triviaqa", ["Parisian", "Paris"]) == [0, 1]   # no partial-word matches


def test_entropy():
    assert discrete_entropy([0] * 10) == 0
    assert abs(discrete_entropy(list(range(10))) - math.log(10)) < 1e-9
    assert abs(discrete_entropy([0] * 5 + [1] * 5) - math.log(2)) < 1e-9
    assert discrete_entropy([]) is None
