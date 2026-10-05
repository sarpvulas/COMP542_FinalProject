"""Build the T5 training table for the instruction-simplification task.

Task (decision, see README): input = one drawing instruction, target = its simplified,
standalone rewrite produced by GPT-4o in the preprocessing notebook.

Input : an Excel file with one row per instruction step and the columns `dataset`,
        `id_of_drawing`, `step_number`, `abstraction_level`, `instructions`, `no_color`
        and `simplified_instructions` (the notebook writes it as `df_no_color.xlsx`
        after merging the GPT-4o output with `merge_simplified`).
Output: the same rows plus four columns read by `T5_Training.py`:

  t5_instr               T5_PROMPT_PREFIX + instructions
  t5_target              simplified_instructions
  t5_instr_no_color      T5_PROMPT_PREFIX + no_color (GPT-4o colour-free instruction)
  t5_target_no_color     simplified_instructions with colour words removed
                         (hexagons_common.strip_colors, a fixed word list, not GPT)

Rows with a missing instruction, no-colour instruction or simplified target are dropped.
"""
import argparse

import pandas as pd

from hexagons_common import build_t5_input, strip_colors

REQUIRED = ["dataset", "id_of_drawing", "step_number", "abstraction_level",
            "instructions", "no_color", "simplified_instructions"]
NEW_COLUMNS = ["t5_instr", "t5_target", "t5_instr_no_color", "t5_target_no_color"]


def build_t5_table(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing columns: {missing}")
    out = df.dropna(subset=["instructions", "no_color", "simplified_instructions"]).copy()
    out = out[out["simplified_instructions"].astype(str).str.strip() != ""]
    out["t5_instr"] = out["instructions"].astype(str).map(build_t5_input)
    out["t5_target"] = out["simplified_instructions"].astype(str)
    out["t5_instr_no_color"] = out["no_color"].astype(str).map(build_t5_input)
    out["t5_target_no_color"] = out["simplified_instructions"].astype(str).map(strip_colors)
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


if __name__ == "__main__":
    main()
