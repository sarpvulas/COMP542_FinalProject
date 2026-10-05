"""CPU tests on tiny synthetic data. Nothing is downloaded and no API is called."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import action_classifier as ac
import board_text as bt
import hexagons_common as hc
import inference
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




class RecordingTokenizer(StubTokenizer):
    """Stub tokenizer that remembers every text it was asked to encode."""
    def __init__(self):
        self.seen = []

    def __call__(self, text, **kw):
        self.seen.append(text)
        return super().__call__(text, **kw)


def board(*filled):
    """A 180-cell board; `filled` is (index, label) pairs."""
    b = [0] * 180
    for i, label in filled:
        b[i] = label
    return b


def raw_df():
    """Two drawings: 1 (train, 2 steps) and 2 (dev, 3 steps), with boards that grow step by step."""
    rows = []
    for drawing, n in [(1, 2), (2, 3)]:
        filled = []
        for step in range(1, n + 1):
            filled.append((step * 7 + drawing, 2 + step))
            rows.append(dict(dataset="train" if drawing == 1 else "dev", id_of_drawing=drawing,
                             step_number=step, abstraction_level="simple",
                             instructions=f"Color the red hexagon {drawing}.{step}",
                             no_color=f"Color the hexagon {drawing}.{step}",
                             resulting_labels=str(board(*filled))))
    return pd.DataFrame(rows)


# ---- board text: rendering, parsing, scoring ------------------------------------------------
def test_render_parse_round_trip_and_format():
    b = board((0, 3), (17, 7), (18, 1), (179, 5))
    text = bt.render_board(b)
    assert text.count(" / ") == 9 and text.split(" / ")[0].startswith("3 0 0")
    assert bt.parse_board(text) == (b, None)
    assert bt.parse_board(f"  {text}\n")[0] == b  # outer whitespace is tolerated


def test_render_rejects_wrong_size():
    with pytest.raises(ValueError):
        bt.render_board([0] * 179)


def test_collapse_board_maps_all_colours_to_filled():
    assert bt.collapse_board([0, 1, 2, 7, 0, 5]) == [0, 1, 1, 1, 0, 1]
    assert set(bt.collapse_board(list(range(8)) * 22 + [0] * 4)) == {0, 1}


@pytest.mark.parametrize("bad", [
    "", "not a board", None, 42,
    " / ".join([" ".join(["0"] * 18)] * 9),                    # 9 rows
    " / ".join([" ".join(["0"] * 17)] * 10),                    # short rows
    " / ".join([" ".join(["0"] * 18)] * 9 + [" ".join(["0"] * 17 + ["8"])]),   # label 8
    " / ".join([" ".join(["0"] * 18)] * 9 + [" ".join(["0"] * 17 + ["x"])]),   # not a number
    " / ".join([" ".join(["0"] * 18)] * 9 + [" ".join(["0"] * 17 + ["-1"])]),  # negative
])
def test_parse_board_reports_malformed_without_raising(bad):
    parsed, reason = bt.parse_board(bad)
    assert parsed is None and isinstance(reason, str) and reason


def test_parse_board_respects_max_label():
    text = bt.render_board(board((3, 2)))
    assert bt.parse_board(text, max_label=1)[0] is None


def test_score_boards_hand_made_case():
    gold = [board((0, 1), (1, 2)), board((5, 3)), board()]
    exact = bt.render_board(gold[0])
    one_wrong = bt.render_board(board((5, 3), (6, 3)))   # 1 extra cell -> 179/180 correct
    result = bt.score_boards([exact, one_wrong, "garbage"], [bt.render_board(g) for g in gold])
    assert result["n"] == 3 and result["n_malformed"] == 1 and list(result["errors"]) == [2]
    assert result["exact_match"] == pytest.approx(1 / 3)
    assert result["cell_accuracy"] == pytest.approx((180 + 179 + 0) / 540)
    # blank board is right on every empty gold cell: 178 + 179 + 180 of 540
    assert result["blank_board_cell_accuracy"] == pytest.approx((178 + 179 + 180) / 540)


def test_score_boards_perfect_and_bad_gold():
    texts = [bt.render_board(board((2, 4)))]
    assert bt.score_boards(texts, texts)["exact_match"] == 1.0
    with pytest.raises(ValueError):
        bt.score_boards(texts, ["bad gold"])
    with pytest.raises(ValueError):
        bt.score_boards(texts, texts + texts)


# ---- T5 data prep, one prompt, one target format --------------------------------------------
def test_prepare_t5_table_columns_history_and_targets():
    out = p5.build_t5_table(raw_df())
    for col in p5.NEW_COLUMNS:
        assert col in out.columns
    d1 = out[out.id_of_drawing == 1].sort_values("step_number")
    assert d1.t5_instr.tolist() == [
        hc.build_t5_input("Color the red hexagon 1.1"),
        hc.build_t5_input("Color the red hexagon 1.1 [SEP] Color the red hexagon 1.2")]
    assert d1.t5_instr_no_color.iloc[1] == hc.build_t5_input("Color the hexagon 1.1 [SEP] Color the hexagon 1.2")
    assert d1.t5_instr.iloc[0].startswith(hc.T5_PROMPT_PREFIX)
    boards = [bt.parse_board(t)[0] for t in d1.resulting_label_list]
    assert boards == [board((8, 3)), board((8, 3), (15, 4))]
    nc = [bt.parse_board(t, max_label=1)[0] for t in d1.resulting_label_list_no_color]
    assert nc == [bt.collapse_board(b) for b in boards]


def test_prepare_without_gpt_column_skips_no_color_input():
    out = p5.build_t5_table(raw_df().drop(columns="no_color"))
    assert "t5_instr_no_color" not in out.columns and "resulting_label_list_no_color" in out.columns


def test_prepare_accepts_list_boards_and_drops_missing_instruction():
    df = raw_df()
    df["resulting_labels"] = df["resulting_labels"].map(eval)
    df.loc[0, "instructions"] = None
    assert len(p5.build_t5_table(df)) == len(df) - 1


def test_prepare_requires_columns_and_unique_index():
    with pytest.raises(ValueError, match="columns"):
        p5.build_t5_table(raw_df().drop(columns="resulting_labels"))
    df = raw_df()
    with pytest.raises(ValueError, match="index"):
        p5.build_t5_table(pd.concat([df, df.iloc[:1]]))


def test_training_and_inference_build_identical_inputs(tmp_path):
    path = tmp_path / "t5.xlsx"
    p5.build_t5_table(raw_df()).to_excel(path)
    table = pd.read_excel(path, index_col=0)
    for no_color, abstraction in [(False, False), (False, True), (True, False)]:
        ds = t5.HexagonsDataset(str(path), "dev", RecordingTokenizer(), max_length=64,
                                include_abstraction_level=abstraction, no_color=no_color)
        train_inputs = []
        for i in range(len(ds)):
            ds[i]
            train_inputs.append(ds.tokenizer.seen[-2])  # input text is encoded before the label
        dev = table[table.dataset == "dev"]
        col = "t5_instr_no_color" if no_color else "t5_instr"
        infer_inputs = inference.build_inputs(dev, col, abstraction and not no_color)
        assert train_inputs == infer_inputs
    assert infer_inputs[0].startswith(hc.T5_PROMPT_PREFIX)
    assert "simplify" not in (PROJECT / "inference.py").read_text()


def test_t5_dataset_target_is_board_text(tmp_path):
    path = tmp_path / "t5.xlsx"
    table = p5.build_t5_table(raw_df())
    table.to_excel(path)
    tok = RecordingTokenizer()
    t5.HexagonsDataset(str(path), "train", tok, max_length=64)[0]
    assert bt.parse_board(tok.seen[-1])[0] is not None          # label text is a valid board text
    nc_tok = RecordingTokenizer()
    t5.HexagonsDataset(str(path), "train", nc_tok, max_length=64, no_color=True)[0]
    assert bt.parse_board(nc_tok.seen[-1], max_label=1)[0] is not None


def test_t5_dataset_masks_padding_in_labels(tmp_path):
    path = tmp_path / "t5.xlsx"
    p5.build_t5_table(raw_df()).to_excel(path)
    item = t5.HexagonsDataset(str(path), "train", StubTokenizer(), max_length=400)[0]
    assert item["labels"][-1] == -100 and item["labels"][0] != -100


def test_inference_scores_malformed_outputs_without_crashing():
    df = p5.build_t5_table(raw_df())
    df["predicted_board_text"] = df["resulting_label_list"]
    df.loc[df.index[1], "predicted_board_text"] = "oops"
    result = inference.score_dataframe(df, "predicted_board_text", "resulting_label_list", 7)
    assert result["n_malformed"] == 1 and list(result["errors"]) == [int(df.index[1])]
    assert result["exact_match"] == pytest.approx((len(df) - 1) / len(df))


def test_notebook_has_no_key_and_uses_env():
    nb = (PROJECT / "Data Preprocess.ipynb").read_text()
    assert "MY_API_KEY" not in nb and "OPENAI_API_KEY" in nb



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
                                    "deberta_abs.py", "lstm_abs.py", "T5_Training.py", "prepare_t5_data.py", "inference.py"])
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
