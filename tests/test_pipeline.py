"""CPU tests on tiny synthetic data. Nothing is downloaded and no API is called."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import action_classifier as ac
import hexagons_common as hc
import prepare_t5_data as p5
import lstm_abs
import T5_Training as t5

PROJECT = Path(__file__).resolve().parents[1] / "Project"


class StubTokenizer:
    """Whitespace tokenizer with fixed-length padding; id 0 = pad, 1 = eos."""
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, max_length=16, padding="max_length",
                 truncation=True, return_tensors="pt"):
        ids = [2 + (sum(map(ord, w)) % 90) for w in str(text).split()][: max_length - 1] + [1]
        mask = [1] * len(ids) + [0] * (max_length - len(ids))
        ids = ids + [0] * (max_length - len(ids))
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}


def raw_df():
    rows = []
    for drawing, n in [(1, 2), (2, 3)]:
        for step in range(1, n + 1):
            rows.append(dict(dataset="train" if drawing == 1 else "dev", id_of_drawing=drawing,
                             step_number=step, abstraction_level="simple",
                             instructions=f"Color the red hexagon {drawing}.{step}",
                             no_color=f"Color the hexagon {drawing}.{step}"))
    return pd.DataFrame(rows)


SEP = f"  {hc.STEP_MARKER}  "


# ---- item 3: merge with length check -------------------------------------------------
def test_merge_aligns_steps_and_reports_mismatch():
    gpt = {1: SEP.join(["a1", "a2"]), 2: SEP.join(["b1", "b2"])}  # drawing 2 has 3 rows
    out, bad = hc.merge_simplified(raw_df(), gpt)
    assert bad == [2]
    assert out.loc[out.id_of_drawing == 1, "simplified_instructions"].tolist() == ["a1", "a2"]
    assert out.loc[out.id_of_drawing == 2, "simplified_instructions"].isna().all()


def test_merge_accepts_lists_and_missing_drawing():
    out, bad = hc.merge_simplified(raw_df(), {1: ["a1", "a2"]})
    assert bad == [2]
    assert out["simplified_instructions"].notna().sum() == 2


def test_merge_tolerates_marker_spacing():
    out, bad = hc.merge_simplified(raw_df(), {1: f"a1 {hc.STEP_MARKER}\n\na2", 2: SEP.join("xyz")})
    assert bad == []
    assert out["simplified_instructions"].tolist() == ["a1", "a2", "x", "y", "z"]


def test_notebook_has_no_key_and_uses_env():
    nb = (PROJECT / "Data Preprocess.ipynb").read_text()
    assert "MY_API_KEY" not in nb and "OPENAI_API_KEY" in nb and "merge_simplified" in nb


# ---- items 1 and 2: T5 data prep and one prompt -----------------------------------------
def test_prepare_t5_table_and_prompt():
    gpt = {1: ["Color Red cell 1", "paint blue cell 2"], 2: ["c1", "c2", "c3"]}
    df, _ = hc.merge_simplified(raw_df(), gpt)
    df.loc[df.index[-1], "simplified_instructions"] = np.nan  # dropped row
    out = p5.build_t5_table(df)
    assert len(out) == 4
    for col in p5.NEW_COLUMNS:
        assert col in out.columns
    first = out.iloc[0]
    assert first.t5_instr == hc.build_t5_input("Color the red hexagon 1.1")
    assert first.t5_instr.startswith(hc.T5_PROMPT_PREFIX)
    assert first.t5_target == "Color Red cell 1"
    assert first.t5_instr_no_color == hc.build_t5_input("Color the hexagon 1.1")
    assert first.t5_target_no_color == "Color cell 1"
    assert out.iloc[1].t5_target_no_color == "paint cell 2"


def test_prepare_requires_columns():
    with pytest.raises(ValueError):
        p5.build_t5_table(raw_df())


@pytest.mark.parametrize("text,expected", [
    ("Make a green and yellow flower", "Make a flower"),
    ("Color white hexagons black", "Color white hexagons"),
    ("Paint the hexagon Red-orange", "Paint the hexagon"),
    ("Fill in red; then blue", "Fill in"),
    ("Make it Navy", "Make it"),
    ("Use red, blue and green here.", "Use here."),
    ("Make the hexagon green instead of red", "Make the hexagon instead of"),  # known ungrammatical case
    ("colour of the sky", "colour of the sky"),
    ("Redo the redwood row", "Redo the redwood row"),
])
def test_strip_colors_examples(text, expected):
    assert hc.strip_colors(text) == expected


def test_merge_rejects_duplicates_and_strips_numbering():
    df = raw_df()
    dup_index = pd.concat([df, df.iloc[:1]])
    with pytest.raises(ValueError, match="unique"):
        hc.merge_simplified(dup_index, {})
    dup_step = pd.concat([df, df.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="step_number"):
        hc.merge_simplified(dup_step, {})
    out, bad = hc.merge_simplified(df, {1: SEP.join(["1. a", "2) b"]), 2: ["Step 1. x", "y 3. z", "3.5 cells"]})
    assert bad == []
    assert out["simplified_instructions"].tolist() == ["a", "b", "x", "y 3. z", "3.5 cells"]


def test_make_abstraction_xlsx_from_tiny_jsonl(tmp_path):
    import json
    import make_abstraction_xlsx as mk
    board = [0] * 180
    (tmp_path / "data").mkdir()
    for split, cat in [("train", "simple"), ("dev", None), ("test", "recursion")]:
        entry = {"index": 7, "drawing_procedure": [[0, "NONE", board], [1, "paint a", board], [2, "paint b", board]]}
        if cat:
            entry["category"] = cat
        (tmp_path / "data" / f"{split}.jsonl").write_text(json.dumps(entry) + "\n")
    df = mk.build_dataframe(tmp_path)
    assert len(df) == 6 and set(df.dataset) == {"train", "dev", "test"}
    assert df.loc[df.dataset == "dev", "abstraction_level"].tolist() == ["NONE", "NONE"]
    out = tmp_path / "df.xlsx"
    df.to_excel(out)
    loaded = hc.load_abstraction_df(out)  # the loader used by lstm_abs.py
    assert loaded["abstraction_label"].tolist() == [0, 0, 2, 2, 3, 3]


def test_inference_uses_shared_prompt():
    src = (PROJECT / "inference.py").read_text()
    assert "build_t5_input" in src and "simplify instructions" not in src


def test_t5_dataset_trains_on_simplified_target(tmp_path):
    gpt = {1: ["Color Red cell 1", "paint blue cell 2"], 2: ["c1", "c2", "c3"]}
    df, _ = hc.merge_simplified(raw_df(), gpt)
    path = tmp_path / "t5.xlsx"
    p5.build_t5_table(df).to_excel(path)
    tok = StubTokenizer()
    ds = t5.HexagonsDataset(str(path), "train", tok, max_length=16)
    item = ds[0]
    expect_in = tok(hc.build_t5_input("Color the red hexagon 1.1"))["input_ids"][0]
    assert torch.equal(item["input_ids"], expect_in)
    target = item["labels"]
    expected = tok("Color Red cell 1")["input_ids"][0]
    assert torch.equal(target[target != -100], expected[expected != 0])
    assert target[-1] == -100  # padding ignored in the loss
    nc = t5.HexagonsDataset(str(path), "train", tok, max_length=16, no_color=True)
    assert nc[0]["labels"][0] != -100


def test_t5_train_and_test_evaluation_run_on_tiny_model():
    from transformers import T5Config, T5ForConditionalGeneration
    torch.manual_seed(0)
    cfg = T5Config(vocab_size=100, d_model=16, d_kv=4, d_ff=32, num_layers=1, num_heads=2,
                   decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    model = T5ForConditionalGeneration(cfg)
    ids = torch.randint(2, 90, (4, 8))
    labels = torch.randint(2, 90, (4, 6))
    labels[:, -1] = -100
    loader = [{"input_ids": ids[:2], "attention_mask": torch.ones(2, 8, dtype=torch.long), "labels": labels[:2]},
              {"input_ids": ids[2:], "attention_mask": torch.ones(2, 8, dtype=torch.long), "labels": labels[2:]}]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    before = t5.evaluate(model, loader, torch.device("cpu"))
    for _ in range(10):
        t5.train_epoch(model, loader, opt, sched, torch.device("cpu"))
    assert t5.evaluate(model, loader, torch.device("cpu")) < before
    assert np.isfinite(t5.evaluate(model, loader, torch.device("cpu"), max_batches=1))


# ---- item 5: state_dict without DataParallel ---------------------------------------------
def test_save_state_dict_with_and_without_dataparallel(tmp_path):
    m = torch.nn.Linear(2, 2)
    ac.save_state_dict(m, tmp_path / "a.pth")
    ac.save_state_dict(torch.nn.DataParallel(m), tmp_path / "b.pth")
    a, b = torch.load(tmp_path / "a.pth"), torch.load(tmp_path / "b.pth")
    assert a.keys() == b.keys() == {"weight", "bias"}


def test_scripts_do_not_use_bare_module_state_dict():
    for name in ["classificationbased.py", "classificationbased_nocolor.py", "classificationbased-abstraction.py"]:
        assert "model.module" not in (PROJECT / name).read_text()


# ---- items 6 and 8: metrics per split, majority baseline, class weights --------------------
def test_split_report_with_baseline():
    df = pd.DataFrame({
        "dataset": ["train"] * 4 + ["dev"] * 4 + ["test"] * 2,
        "action_label": [0, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        "predictions": [np.nan] * 4 + [0, 0, 1, 0, 0, 1],
    })
    rep = ac.split_report(df, "action_label")
    assert set(rep) == {"dev", "test"}
    assert rep["dev"]["model"]["accuracy"] == 0.75
    assert rep["dev"]["majority_baseline"]["accuracy"] == 0.5  # predicts 0 (train majority)
    assert rep["dev"]["majority_baseline"]["macro_f1"] == pytest.approx(1 / 3)
    assert rep["test"]["model"] == {"n": 2, "accuracy": 1.0, "macro_f1": 1.0}


def test_class_weights_upweight_rare_class_and_zero_absent():
    w = hc.class_weights([0, 0, 0, 1], 3)
    assert w[1] > w[0] > 0 and w[2] == 0
    assert w[1] == pytest.approx(3 * w[0])


# ---- item 7: seeds ------------------------------------------------------------------------
def test_set_seed_reproducible():
    hc.set_seed(7); a = (torch.rand(3), np.random.rand(), __import__("random").random())
    hc.set_seed(7); b = (torch.rand(3), np.random.rand(), __import__("random").random())
    assert torch.equal(a[0], b[0]) and a[1:] == b[1:]


@pytest.mark.parametrize("script", ["classificationbased.py", "classificationbased_nocolor.py",
                                    "classificationbased-abstraction.py", "classification_evaluation.py",
                                    "deberta_abs.py", "lstm_abs.py", "T5_Training.py", "prepare_t5_data.py"])
def test_cli_help_lists_seed_and_file_args(script):
    out = subprocess.run([sys.executable, str(PROJECT / script), "--help"], capture_output=True, text=True,
                         cwd=PROJECT)
    assert out.returncode == 0, out.stderr
    if script not in ("prepare_t5_data.py",):
        assert "--seed" in out.stdout
    if script != "T5_Training.py":
        assert "--input_file" in out.stdout


def test_default_file_names_unchanged():
    for variant, cfg in ac.VARIANTS.items():
        assert cfg["model_path"].endswith(".pth")
    import argparse
    p = argparse.ArgumentParser(); ac._add_common_args(p, "color")
    assert p.parse_args([]).input_file == "expanded_df_final.xlsx"
    assert p.parse_args([]).model_path == "model.pth"


# ---- classifier training end to end on a tiny DeBERTa -------------------------------------
def tiny_deberta(num_labels):
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
    cfg = DebertaV2Config(vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                          intermediate_size=32, max_position_embeddings=32, relative_attention=True,
                          position_buckets=8, max_relative_positions=8, pos_att_type=["p2c", "c2p"],
                          num_labels=num_labels)
    return DebertaV2ForSequenceClassification(cfg)


@pytest.mark.parametrize("weighted", [False, True])
def test_fit_and_predict_tiny_model(tmp_path, weighted):
    rows = []
    for split, n in [("train", 12), ("dev", 4)]:
        for i in range(n):
            rows.append(dict(dataset=split, final_input=f"{i % 3} {i % 5} go", action_label=int(i % 3 == 0)))
    df = pd.DataFrame(rows)
    ds = ac.HexagonsDataset(df, StubTokenizer(), "final_input", "action_label", max_len=8)
    assert ds[0]["input_ids"].shape == (8,)
    model = tiny_deberta(8)
    out = tmp_path / "m.pth"
    orig = ac.HexagonsDataset
    ac.HexagonsDataset = lambda d, t, tc, lc: orig(d, t, tc, lc, max_len=8)
    try:
        ac.fit(model, StubTokenizer(), df, "color", torch.device("cpu"), epochs=1, batch_size=4,
               class_weighted=weighted, model_path=str(out), seed=1)
    finally:
        ac.HexagonsDataset = orig
    assert out.exists()
    loader = torch.utils.data.DataLoader(ds, batch_size=5, shuffle=False)
    preds = ac.predict(model, loader, torch.device("cpu"))
    assert len(preds) == len(df) and set(preds) <= set(range(8))


# ---- item 9: BiLSTM padding invariance ----------------------------------------------------
def make_lstm():
    torch.manual_seed(0)
    return lstm_abs.BiLSTM(vocab_size=50, embedding_dim=8, hidden_dim=6, output_dim=4, num_layers=2,
                           bidirectional=True, dropout=0.5).eval()


def padded(tokens, length):
    ids = torch.tensor([tokens + [0] * (length - len(tokens))])
    mask = torch.tensor([[1] * len(tokens) + [0] * (length - len(tokens))])
    return ids, mask


def test_bilstm_output_independent_of_padding_with_mask():
    model = make_lstm()
    tokens = [5, 9, 3, 12]
    outs = [model(*padded(tokens, n)) for n in (4, 10, 40)]
    assert torch.allclose(outs[0], outs[1], atol=1e-5) and torch.allclose(outs[0], outs[2], atol=1e-5)


def test_bilstm_without_mask_is_padding_dependent():
    model = make_lstm()
    tokens = [5, 9, 3, 12]
    a, b = model(padded(tokens, 4)[0]), model(padded(tokens, 40)[0])
    assert not torch.allclose(a, b, atol=1e-5)  # the behaviour the mask fixes


def test_bilstm_mixed_batch_matches_single():
    model = make_lstm()
    ids = torch.tensor([[5, 9, 3, 12, 0, 0], [7, 2, 0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]])
    both = model(ids, mask)
    assert torch.allclose(both[1], model(*padded([7, 2], 2))[0], atol=1e-5)
    assert torch.allclose(both[0], model(*padded([5, 9, 3, 12], 4))[0], atol=1e-5)
