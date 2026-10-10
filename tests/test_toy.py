"""Toy rewards (planted effect + null control): known answers, by hand, no measured numbers."""
from loopgym import env
from loopgym.env import toy
from loopgym.env.toy import grade

BANANA = toy.load_toy("banana")
COINFLIP = toy.load_toy("coinflip")


def test_banana_present_scores_one():
    assert grade(BANANA, BANANA.parse("I like Banana"))[0] == 1.0


def test_banana_absent_scores_zero():
    assert grade(BANANA, BANANA.parse("apple"))[0] == 0.0


def test_banana_case_insensitive():
    assert grade(BANANA, BANANA.parse("a BANANA!"))[0] == 1.0


def test_banana_in_window_scores_one():
    assert grade(BANANA, BANANA.parse("fig fig banana"))[0] == 1.0


def test_banana_beyond_20_tokens_scores_zero():
    # 21 whole words before "banana": each word is at least one token, so banana
    # starts at token >= 21, outside the first-20 window, however "fig " splits.
    assert grade(BANANA, BANANA.parse("fig " * 21 + "banana"))[0] == 0.0


def test_banana_reference_one_stub_zero():
    assert env.grade(BANANA, BANANA.reference())[0] == 1.0
    assert env.grade(BANANA, BANANA.stub())[0] == 0.0


def _mean10k(text):
    return sum(grade(COINFLIP, COINFLIP.parse(text), step=s, index=i)[0]
               for s in range(100) for i in range(100)) / 10000


def test_coinflip_mean_over_10k_is_half():
    assert abs(_mean10k("banana") - 0.5) <= 0.02


def test_coinflip_independent_of_text():
    for s in range(50):
        for i in range(50):
            assert grade(COINFLIP, {"reply.txt": "apple"}, step=s, index=i)[0] == \
                grade(COINFLIP, {"reply.txt": "banana"}, step=s, index=i)[0]
    assert abs(_mean10k("apple") - 0.5) <= 0.02


def test_coinflip_deterministic_binary():
    assert grade(COINFLIP, {"reply.txt": "x"}, step=7, index=3)[0] == \
        grade(COINFLIP, {"reply.txt": "y"}, step=7, index=3)[0]
    vals = {grade(COINFLIP, {"reply.txt": "x"}, step=7, index=i)[0] for i in range(100)}
    assert vals == {0.0, 1.0}  # not a constant: both outcomes occur


def test_prompt_and_parse_interface():
    assert BANANA.prompt() == [{"role": "user", "content": "Name a berry."}]
    assert BANANA.parse("hello") == {"reply.txt": "hello"}
    assert COINFLIP.prompt() == BANANA.prompt()


def test_grade_returns_reward_and_info():
    r, info = grade(BANANA, BANANA.parse("banana"))
    assert r == info["reward"] == 1.0
    assert info["runner"] == "toy"


def test_unknown_toy_name_raises():
    try:
        toy.load_toy("nope")
    except ValueError:
        return
    raise AssertionError("load_toy('nope') should raise ValueError")
