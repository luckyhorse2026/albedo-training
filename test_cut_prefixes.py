from cut_prefixes import (
    cut_prefix,
    fake_row,
    first_edit_index,
    prefix_phase,
    turn_idx_for,
)


def test_first_edit_is_assistant_five():
    assert first_edit_index(fake_row()) == 5


def test_cold_has_one_or_two_assistants_and_no_edit():
    messages = fake_row()
    for pick, want_asst in ((1, 1), (2, 2)):
        idx = turn_idx_for("cold", 5, cold_pick=pick)
        prefix = cut_prefix(messages, idx)
        assert prefix is not None
        asst = [m for m in prefix if m["role"] == "assistant"]
        assert len(asst) == want_asst
        assert prefix_phase(prefix) == "cold"
        assert "sed -i" not in "".join(m["content"] for m in asst)


def test_pre_edit_has_no_sed_but_more_than_two_assistants():
    prefix = cut_prefix(fake_row(), turn_idx_for("pre_edit", 5, cold_pick=1))
    assert prefix is not None
    asst = [m for m in prefix if m["role"] == "assistant"]
    assert len(asst) == 3
    assert prefix_phase(prefix) == "pre_edit"
    assert "sed -i" not in "".join(m["content"] for m in asst)


def test_at_edit_includes_the_sed():
    prefix = cut_prefix(fake_row(), turn_idx_for("at_edit", 5, cold_pick=1))
    assert prefix is not None
    asst = [m for m in prefix if m["role"] == "assistant"]
    assert len(asst) == 5
    assert prefix_phase(prefix) == "at_edit"
    assert "sed -i" in asst[-1]["content"]
