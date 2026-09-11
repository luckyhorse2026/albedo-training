from cut_prefixes import demo_prefix
from filter_rollouts import judge
from roll_king import fake_king_skip_verify, fake_king_verify, roll_one, stub_observe


def _roll(king):
    return roll_one(
        demo_prefix("cold"), rollout=1, n=2, king=king, observe=stub_observe, horizon=4
    )


def test_verify_is_kept():
    rec = judge(_roll(fake_king_verify))
    assert rec["keep"] is True
    assert rec["reasons"] == []


def test_skip_verify_is_dropped():
    rec = judge(_roll(fake_king_skip_verify))
    assert rec["keep"] is False
    assert "no_verify_after_edit" in rec["reasons"]
