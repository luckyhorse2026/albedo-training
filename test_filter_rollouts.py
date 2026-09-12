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


def test_search_spam_on_pre_edit():
    prefix = demo_prefix("pre_edit")

    def king(_messages, step):
        if step == 0:
            return "THOUGHT: look around\n\n```bash\nfind . -name '*.py'\n```"
        if step == 1:
            return "THOUGHT: more\n\n```bash\ngrep -rn add .\n```"
        return "THOUGHT: still\n\n```bash\nfind . -type f\n```"

    rec = judge(roll_one(prefix, rollout=1, n=1, king=king, observe=stub_observe, horizon=3))
    assert rec["keep"] is False
    assert "search_spam" in rec["reasons"]


def test_infra_thrash():
    prefix = demo_prefix("cold")

    def king(_messages, step):
        if step == 0:
            return "THOUGHT: need pytest\n\n```bash\npip install pytest\n```"
        return "THOUGHT: where is it\n\n```bash\nwhich pytest\n```"

    rec = judge(roll_one(prefix, rollout=1, n=1, king=king, observe=stub_observe, horizon=2))
    assert rec["keep"] is False
    assert "infra_thrash" in rec["reasons"]
