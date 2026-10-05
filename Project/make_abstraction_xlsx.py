"""Build the input Excel file for `lstm_abs.py` / `deberta_abs.py` from the public Hexagons data.

No GPT and no API key needed: the abstraction classifiers read only the columns
`instructions`, `abstraction_level` and `dataset`. The flattening is the notebook's
`process_file` logic (one row per drawing step, step 0 skipped). The first column of the
Excel file is the row index, which the scripts load with `index_col=0`.

    python make_abstraction_xlsx.py --hexagons_dir path/to/Hexagons --output_file df_no_color.xlsx

The notebook's file `df_no_color.xlsx` also has the GPT column `no_color`; the abstraction
scripts do not use it. The output is derived from the Hexagons data (research and academic use
only, CC-BY 4.0): do not commit or redistribute it.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def process_file(file_path, dataset_type):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            index = entry['index']  # ID of the drawing
            category = entry.get('category', 'NONE')  # Abstraction level

            for step_id, instruction, board_state in entry['drawing_procedure']:
                if step_id == 0:
                    continue  # initial blank board, no instruction
                data.append({
                    'id_of_drawing': index,
                    'step_number': step_id,
                    'instructions': instruction,
                    'abstraction_level': category,
                    'resulting_labels': board_state,
                    'dataset': dataset_type,
                })
    return data


def build_dataframe(hexagons_dir):
    rows = []
    for split in ("train", "dev", "test"):
        rows += process_file(Path(hexagons_dir) / "data" / f"{split}.jsonl", split)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hexagons_dir", required=True, help="Clone of the Hexagons repository.")
    parser.add_argument("--output_file", default="df_no_color.xlsx")
    args = parser.parse_args()
    df = build_dataframe(args.hexagons_dir)
    df.to_excel(args.output_file)
    print(df["dataset"].value_counts().to_string())
    print(f"{len(df)} rows -> {args.output_file}")


if __name__ == "__main__":
    main()
