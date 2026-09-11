from cut_prefixes import demo_prefix
from format_turn import first_bash, is_submit
from roll_king import (
    fake_king_skip_verify,
    fake_king_verify,
    roll_one,
    stub_observe,
)


def test_verify_path_edits_then_checks_then_submits():
    rec = roll_one(
        demo_prefix("cold"),
        rollout=1,
        n=2,
        king=fake_king_verify,
        observe=stub_observe,
        horizon=4,
    )
    assert rec["stopped"] == "submit"
    assert "sed -i" in rec["commands"][0]
    assert "python -c" in rec["commands"][1]
    assert is_submit(rec["commands"][2])


def test_skip_path_never_checks():
    rec = roll_one(
        demo_prefix("cold"),
        rollout=2,
        n=2,
        king=fake_king_skip_verify,
        observe=stub_observe,
        horizon=4,
    )
    assert rec["stopped"] == "horizon"
    assert any("sed -i" in c for c in rec["commands"])
    assert not any("python -c" in c for c in rec["commands"])
    assert first_bash(rec["continuation"][0]["content"])
