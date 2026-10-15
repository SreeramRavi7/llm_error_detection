from src.grading import grade, normalize, extract_number


def test_normalize():
    assert normalize("The Eiffel Tower!") == "eiffel tower"


def test_triviaqa_exact_and_alias():
    gold = ["Paris", "Paris, France"]
    assert grade("triviaqa", "Paris", gold) == 1
    assert grade("triviaqa", "It's Paris.", gold) == 1
    assert grade("triviaqa", "paris france", gold) == 1


def test_triviaqa_wrong_and_partial_word():
    gold = ["Paris"]
    assert grade("triviaqa", "London", gold) == 0
    assert grade("triviaqa", "Parisian", gold) == 0   # no substring false positives
    assert grade("triviaqa", "", gold) == 0


def test_gsm8k_numbers():
    assert extract_number("so the total is 1,250 dollars.\n#### 1,250") == 1250
    assert grade("gsm8k", "Step 1... #### 42", ["42"]) == 1
    assert grade("gsm8k", "The answer is 42.", ["42"]) == 1
    assert grade("gsm8k", "#### 41", ["42"]) == 0
    assert grade("gsm8k", "#### 3.50", ["3.5"]) == 1
    assert grade("gsm8k", "no number here", ["42"]) == 0
    assert grade("gsm8k", "#### -7", ["-7"]) == 1
