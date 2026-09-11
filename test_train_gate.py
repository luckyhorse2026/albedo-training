from pathlib import Path

from filter_rollouts import judge
from gate import score_rollouts
from pack_pairs import demo_judged
from train import commands, recipe


def test_recipe_order_and_refusal_fields():
    r = recipe(king_path="/king", sft_path="out/sft.jsonl", dpo_path="out/dpo.jsonl", out_dir="out/train")
    assert r["order"] == ["sft", "dpo"]
    assert r["sft"]["lr"] == 5e-6
    assert any("n_prefix" in r["sft"]["mask"] for _ in [1])
    assert commands(r)


def test_gate_on_demo_pair(tmp_path: Path):
    rows = demo_judged()
    report = score_rollouts(rows)
    assert report["n"]["cold"] == 2
    assert report["cold_edit_rate"] == 1.0
    assert report["cold_verify_rate"] == 0.5
    assert report["loop_rate"] == 0.5
