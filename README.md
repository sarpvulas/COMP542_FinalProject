# COMP 542 Hexagons Project

Course project (Koç University, COMP 542): grounding natural-language drawing instructions on a hexagonal board, using T5 and DeBERTa on the Hexagons dataset.

## TL;DR

The Hexagons dataset pairs human-written instructions with the board state they produce on an 18 x 10 hexagon grid, at different levels of abstraction. This repository holds the code for a data-preparation notebook (GPT-4o is used to strip colour words and to simplify instructions), a T5 training and inference script, and DeBERTa / BiLSTM classifiers that predict per-cell actions and the abstraction level of an instruction. No results are recorded in this repository, so no headline numbers are claimed.

## Results

TODO(sarp): add results table (accuracy / macro-F1 per model and split; the scripts print them but no outputs or logs were committed).

The only numbers in the repo are dataset statistics, printed by `Project/Data Preprocess.ipynb`: 4,177 instruction steps (train 3,278, dev 446, test 453).

## Data

[Hexagons](https://github.com/OnlpLab/Hexagons) (OnlpLab). Each entry is a drawing procedure: a sequence of instructions, each with the resulting board state (180 cells = 10 rows x 18 columns, integer colour labels), plus an abstraction category (`simple`, `symmetry`, `composed objects`, `conditions`, `bounded iteration`, `conditional iteration`, `recursion`, `other`, or none). The dataset is not included here; the notebook expects `Hexagons/data/{train,dev,test}.jsonl`.

Citation:

> Royi Lachmy, Valentina Pyatkin, Avshalom Manevich, Reut Tsarfaty. *Draw Me a Flower: Processing and Grounding Abstraction in Natural Language.* Transactions of the ACL, 2022. [arXiv:2106.14321](https://arxiv.org/abs/2106.14321)

The dataset repository is MIT-licensed and its page states CC-BY 4.0 for its resources; check its terms before redistributing any derived files.

## Pipeline

| Step | File | What it does |
|------|------|--------------|
| 1. Preprocess | `Project/Data Preprocess.ipynb` | Flattens the jsonl files into one row per instruction step; asks GPT-4o for colour-free and for simplified versions of each drawing's instructions; builds per-cell rows (`row_number`, `column_number`, `action_label`) with the instruction history of the drawing as input; writes `expanded_df_final.xlsx`. |
| 2. T5 | `Project/T5_Training.py`, `Project/inference.py` | Fine-tunes `google/t5-v1_1-base` on the Excel table (AdamW, lr 3e-5, default 50 epochs, keeps the best-dev-loss checkpoint). Inference writes a `simplified_instructions` column. |
| 3. Action classifiers | `classificationbased.py`, `classificationbased_nocolor.py`, `classificationbased-abstraction.py`, `classification_evaluation.py` | Fine-tune `microsoft/mdeberta-v3-base` (20 epochs, Adam, lr 3e-5, batch 25) to predict, for a given cell, the colour label painted at this step (8 classes), a coloured / not-coloured flag (2 classes, no colour words), or the same with the abstraction level in the input. The evaluation script reloads `model.pth` and writes predictions for all rows to `predictiondf.xlsx`. |
| 4. Abstraction level | `deberta_abs.py`, `lstm_abs.py` | Classify an instruction into 4 abstraction groups (simple; symmetry/other; composed objects/conditions/none; iteration/recursion) with mDeBERTa and with a 2-layer BiLSTM. Print accuracy and macro-F1 per split. |

All scripts are in `Project/` and read their Excel inputs from the current directory.

## Tech stack

Python, PyTorch, Hugging Face Transformers, pandas, scikit-learn, OpenAI API (notebook only).

## Quickstart

```bash
pip install -r requirements.txt
```

`requirements.txt` is unpinned; the imports and syntax of all scripts were checked on Python 3.12 with current releases (torch 2.14, transformers 5.18, pandas 3.0). Training was not run for this README (no dataset, no GPU, and models are large downloads), so the commands below are the authors' usage, not re-verified:

```bash
cd Project
# 1. run Data Preprocess.ipynb (needs the Hexagons data and your own OpenAI key in the client setup cell)
python T5_Training.py --data_file path/to/dataset.xlsx --include_abstraction_level --no_color --epochs 50
python inference.py --device cuda --model_path path/to/model --batch_size 100 --show_time_remaining --input_file in.xlsx --output_file out.xlsx
python classificationbased.py          # also classificationbased_nocolor.py, classificationbased-abstraction.py
python classification_evaluation.py
python deberta_abs.py
python lstm_abs.py
```

T5 arguments: `--data_file` (required), `--include_abstraction_level`, `--no_color`, `--epochs` (default 50). Inference arguments: `--device`, `--model_path`, `--batch_size`, `--show_time_remaining`, `--input_file`, `--output_file`.

## Reproducibility notes

- No random seeds are set anywhere; runs are not deterministic.
- The OpenAI calls use `gpt-4o` and the notebook's API key is a placeholder (`MY_API_KEY`); GPT outputs were saved as pickles that are not committed, and several drawings were repaired by hand in the notebook (cell "new_instructions_clean").
- Intermediate files (`df_no_color.xlsx`, `expanded_df_final.xlsx`, `model*.pth`) are not in the repo and are git-ignored.

## Limitations

- Course project, not a maintained library. No tests, no CI, no committed results or logs.
- The T5 scripts expect columns (`t5_instr`, `resulting_label_list`, `t5_instr_no_color`, `resulting_label_list_no_color`) that the notebook never creates, and `inference.py` prompts with `simplify instructions: ...` while training uses a different input format. The T5 data-prep step is therefore missing from the repo.
- The notebook creates the simplified instructions with GPT-4o but never merges them into the dataframe.
- `T5_Training.py` builds the test loader but never evaluates on it. The classifier scripts train on the train split and report only losses; `classification_evaluation.py` predicts over all splits and computes no metric.
- The three `classificationbased*.py` scripts save with `model.module.state_dict()`, which only works when `DataParallel` is active (more than one GPU); on one GPU or CPU they fail at the end of the first epoch.
- Input paths are relative file names (for example `expanded_df_final.xlsx`) with no CLI option, except in the T5 scripts.
- `deberta_abs.py` had undefined names (`valid_df`, `np`, metric imports); fixed in this version but not run.

## Credits and license

Course project for COMP 542 at Koç University by Sarp Vulaş and Egecan Esen. Dataset and task: Lachmy et al. (see above). Code licensed under Apache-2.0 (see `LICENSE`).

Author: Sarp Vulaş, Dubai, MSc Computational Finance, King's College London. LinkedIn: TODO(sarp): add LinkedIn URL.
