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


def test_gsm8k_final_answer_formats():
    # real pilot endings that the old last-number rule got wrong
    assert grade("gsm8k", "\\[ x = 60 \\] So, Jason should add **60 grams** when using 900 ml.", ["60"]) == 1
    assert grade("gsm8k", "37 x 11 = 407. Ms. Alice can grade **407** papers in 11 hours.", ["407"]) == 1
    assert grade("gsm8k", "The final answer is: \\[ \\boxed{87} \\] after 3 steps", ["87"]) == 1
    assert grade("gsm8k", "total is $\\boxed{1552.50}$.", ["15525"]) == 0
    assert grade("gsm8k", "Final Answer: #### 16000", ["5,600"]) == 0
    assert grade("gsm8k", "**Step 3:** add them\n4 + 8 = 12", ["12"]) == 1   # bold header, not the answer


def test_gsm8k_unboxed_last_sentence():
    assert grade("gsm8k", "work...\nSo, Ginger needs 650 straws to make 10 mats.", ["650"]) == 1
    assert grade("gsm8k", "work...\nSo, Gail has 2 $10 bills in her wallet.", ["2"]) == 1
    assert grade("gsm8k", "Total is 17. So, there are 10 boys and 7 girls left.", ["17"]) == 0   # only last sentence counts
    assert grade("gsm8k", "the answer is \\boxed{8} cans in 480 minutes", ["480"]) == 0          # boxed is strict
