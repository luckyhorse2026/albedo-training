from pack_pairs import demo_judged, pack


def test_one_sft_and_one_dpo_from_fake_pair():
    sft, dpo = pack(demo_judged())
    assert len(sft) == 1
    assert len(dpo) == 1
    assert sft[0]["sample_id"] == dpo[0]["sample_id"]
    assert sft[0]["n_prefix"] < len(sft[0]["messages"])
    assert "python -c" in str(dpo[0]["chosen"])
    assert "grep" in str(dpo[0]["rejected"])
    assert dpo[0]["prompt"] == sft[0]["messages"][: sft[0]["n_prefix"]]
