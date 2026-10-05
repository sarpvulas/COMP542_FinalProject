"""Build the T5 training table for board prediction (the paper's instruction-to-execution task).

Task: from the instructions of a drawing up to step t, T5 writes the board after step t.

Input : an Excel file with one row per instruction step and the columns `dataset`,
        `id_of_drawing`, `step_number`, `abstraction_level`, `instructions`,
        `resulting_labels` (the 180 board labels, stored by Excel as the string "[0, 1, ...]"),
        and optionally `no_color` (the GPT-4o colour-free instruction from the notebook).
        `make_abstraction_xlsx.py` writes such a file without GPT (no `no_color` column);
        the notebook's `df_no_color.xlsx` has all columns.
Output: the same rows plus the columns read by `T5_Training.py` and `inference.py`:

  t5_instr                       T5_PROMPT_PREFIX + the drawing's instructions up to this step,
                                 joined with " [SEP]" (the same history the classifiers use)
  resulting_label_list           the board after this step as text (board_text.render_board)
  t5_instr_no_color              as t5_instr, from the `no_color` instructions (only if that column exists)
  resulting_label_list_no_color  the board with every paint colour collapsed to 1 (board_text.collapse_board)

Assumption, stated plainly: the original T5 code never fed a board into the model (only
`t5_instr` and, optionally, the abstraction level), so the previous board is not part of the
input; the instruction history carries it, as for the classifiers.
"""
import argparse
import ast

import pandas as pd

from board_text import collapse_board, render_board
from hexagons_common import build_t5_input

REQUIRED = ["dataset", "id_of_drawing", "step_number", "abstraction_level", "instructions", "resulting_labels"]
HISTORY_SEP = " [SEP] "
NEW_COLUMNS = ["t5_instr", "resulting_label_list", "t5_instr_no_color", "resulting_label_list_no_color"]


def _labels(value):
    return ast.literal_eval(value) if isinstance(value, str) else list(value)


def _history(df, column):
    """For every row, the `column` texts of its drawing up to and including its step, joined."""
    out = pd.Series(index=df.index, dtype="object")
    for _, group in df.groupby("id_of_drawing", sort=False):
        ordered = group.sort_values("step_number")
        texts = ordered[column].astype(str).tolist()
        for i, idx in enumerate(ordered.index):
            out.loc[idx] = HISTORY_SEP.join(texts[: i + 1])
    return out


def build_t5_table(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing columns: {missing}")
    if not df.index.is_unique:
        raise ValueError("input needs a unique index")
    out = df.dropna(subset=["instructions"]).copy()  # 1 row in Hexagons has no instruction
    out["t5_instr"] = _history(out, "instructions").map(build_t5_input)
    boards = out["resulting_labels"].map(_labels)
    out["resulting_label_list"] = boards.map(render_board)
    out["resulting_label_list_no_color"] = boards.map(lambda b: render_board(collapse_board(b)))
    if "no_color" in out.columns:
        out = out.dropna(subset=["no_color"]).copy()
        out["t5_instr_no_color"] = _history(out, "no_color").map(build_t5_input)
    return out.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_file", default="df_no_color.xlsx")
    parser.add_argument("--output_file", default="t5_data.xlsx")
    args = parser.parse_args()
    df = pd.read_excel(args.input_file, index_col=0)
    out = build_t5_table(df)
    out.to_excel(args.output_file)
    print(f"{len(df)} rows in, {len(out)} rows out -> {args.output_file}")
    print(out["dataset"].value_counts().to_string())
    if "t5_instr_no_color" not in out.columns:
        print("no `no_color` column: t5_instr_no_color not written (T5_Training --no_color needs it)")


if __name__ == "__main__":
    main()
