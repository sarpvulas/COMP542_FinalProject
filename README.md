# COMP 542 Hexagons Project

Course project (Koç University, COMP 542): grounding natural-language drawing instructions on a hexagonal board, using T5 and DeBERTa on the Hexagons dataset.

## TL;DR

The Hexagons dataset pairs human-written instructions with the board state they produce on an 18 x 10 hexagon grid, at different levels of abstraction. This repository holds the code for a data-preparation notebook (GPT-4o is used to strip colour words and to simplify instructions), a T5 instruction-simplification pipeline (data prep, training, inference), and DeBERTa / BiLSTM classifiers that predict per-cell actions and the abstraction level of an instruction. The only results recorded are limited CPU runs of the BiLSTM abstraction classifier (see Results); the DeBERTa and T5 models have not been run here.

## Results

These are **limited CPU runs**, not tuned and not the project's final models. Hardware: Apple M3 Max, CPU only, PyTorch 2.10, Transformers 5.2. Seed 42, 20 epochs, batch 256, final-epoch weights (no early stopping). Input: the Hexagons jsonl files flattened as in the notebook's `process_file` (no GPT columns are needed by the BiLSTM); 4,177 steps (train 3,278, dev 446, test 453), of which 1 train row with a missing instruction is dropped by the script (train n = 3,277). Four abstraction groups: 0 simple; 1 symmetry + other; 2 composed objects + conditions + `NONE`; 3 bounded / conditional iteration + recursion. The groups are mixed: `NONE` marks second-round drawings that have no category (1,223 of 4,176 rows, about 29%), and group 2 is the majority class (1,509 of 3,277 train rows, about 46%). The classifier therefore partly separates "no category" from the others and does not measure abstraction alone. Also, 22 of 446 dev and 18 of 453 test instructions appear verbatim in train (short texts), so some overlap is expected.

Reproduce without any API key. `make_abstraction_xlsx.py` builds the input (columns `instructions`, `abstraction_level`, `dataset`, and the row index in the first column, which the loader reads with `index_col=0`) from a clone of the public Hexagons repository, using the notebook's `process_file` logic; the file is derived from the dataset, so do not commit it. Commands, from `Project/`:

```bash
git clone https://github.com/OnlpLab/Hexagons.git
python make_abstraction_xlsx.py --hexagons_dir Hexagons --output_file df_no_color.xlsx
python lstm_abs.py --seed 42 --epochs 20                    # run 1, 7 min 44 s wall clock
python lstm_abs.py --seed 42 --epochs 20 --class_weighted   # run 2, 8 min 41 s wall clock
```

BiLSTM abstraction classifier, dev / test, with the majority-class baseline (always predict the most frequent train group):

| Run | Dev accuracy | Dev macro-F1 | Test accuracy | Test macro-F1 |
|-----|--------------|--------------|---------------|---------------|
| Majority-class baseline | 0.5269 | 0.1725 | 0.5298 | 0.1732 |
| BiLSTM, unweighted loss | 0.5381 | 0.4397 | 0.5342 | 0.3516 |
| BiLSTM, `--class_weighted` | 0.4709 | 0.3979 | 0.4503 | 0.3734 |

Reading: the BiLSTM barely beats the majority baseline on accuracy but clearly beats it on macro-F1. The training accuracy at the last epoch (0.84 unweighted) against dev (0.54) shows overfitting, and dev loss rose from epoch 2 on; one seed, no variance estimate. Class weighting lowers accuracy and gives a similar macro-F1 (higher on test, lower on dev); with one seed this difference is not established. The unweighted run was reproduced exactly by an independent run; the class-weighted numbers were produced once, with one seed, and nobody has re-run them independently.

Only the BiLSTM abstraction classifier was run on real data. T5: `prepare_t5_data.py` and `T5_Training.py` ran one step end to end with `t5-small` on CPU (`--epochs 1 --max_batches 1 --batch_size 2 --max_length 64`), on the real instructions with the instruction itself as a placeholder target, to check that the code runs (test loss 0.54 on one batch). That loss is a smoke check, not a result.

TODO(sarp): DeBERTa results (action classifiers and `deberta_abs.py`) and T5 results with the real GPT-4o targets, to be run on a GPU.

## Data

[Hexagons](https://github.com/OnlpLab/Hexagons) (OnlpLab). Each entry is a drawing procedure: a sequence of instructions, each with the resulting board state (180 cells = 10 rows x 18 columns, integer colour labels), plus an abstraction category (`simple`, `symmetry`, `composed objects`, `conditions`, `bounded iteration`, `conditional iteration`, `recursion`, `other`, or none). The dataset is not included here; the notebook expects `Hexagons/data/{train,dev,test}.jsonl`.

Citation:

> Royi Lachmy, Valentina Pyatkin, Avshalom Manevich, Reut Tsarfaty. *Draw Me a Flower: Processing and Grounding Abstraction in Natural Language.* Transactions of the ACL, 2022. [arXiv:2106.14321](https://arxiv.org/abs/2106.14321)

The dataset repository carries an MIT LICENSE file, and its README states CC-BY 4.0 for its resources and restricts use to research and academic purposes. This repository does not contain the dataset or any file derived from it (no data, predictions, or weights are committed; `*.xlsx`, `*.pth` and the T5 checkpoints are git-ignored).

## Pipeline

| Step | File | What it does |
|------|------|--------------|
| 1. Preprocess | `Project/Data Preprocess.ipynb` | Flattens the jsonl files into one row per instruction step; asks GPT-4o for colour-free and for simplified versions of each drawing's instructions; builds per-cell rows (`row_number`, `column_number`, `action_label`) with the instruction history of the drawing as input; writes `expanded_df_final.xlsx`. |
| 2. T5 | `Project/prepare_t5_data.py`, `Project/T5_Training.py`, `Project/inference.py` | Task: **instruction simplification** (decision below). `prepare_t5_data.py` builds `t5_instr`, `t5_target`, `t5_instr_no_color`, `t5_target_no_color`; `T5_Training.py` fine-tunes `google/t5-v1_1-base` on them (AdamW, lr 3e-5, default 50 epochs, keeps the best-dev-loss checkpoint, evaluates that checkpoint once on the test split); `inference.py` writes a `simplified_instructions` column. All three build the input with `hexagons_common.build_t5_input`. |
| 3. Action classifiers | `classificationbased.py`, `classificationbased_nocolor.py`, `classificationbased-abstraction.py`, `classification_evaluation.py` | Fine-tune `microsoft/mdeberta-v3-base` (20 epochs, Adam, lr 3e-5, batch 25) to predict, for a given cell, the colour label painted at this step (8 classes), a coloured / not-coloured flag (2 classes, no colour words), or the same with the abstraction level in the input. `classification_evaluation.py` reloads the weights, predicts the requested splits (default dev and test), writes `predictiondf.xlsx`, and prints and saves accuracy and macro-F1 per split next to the majority-class baseline (`metrics_<variant>.json`). |
| 4. Abstraction level | `deberta_abs.py`, `lstm_abs.py` | Classify an instruction into 4 abstraction groups (simple; symmetry/other; composed objects/conditions/none; iteration/recursion) with mDeBERTa and with a 2-layer BiLSTM. Print accuracy and macro-F1 per split, with the majority-class baseline. The BiLSTM packs padded sequences, so padding does not change its output. |

All scripts are in `Project/`; shared helpers (seed, T5 prompt, GPT merge, metrics, class weights) are in `Project/hexagons_common.py`. Excel inputs and outputs default to the old file names, in the current directory, and can be changed with `--input_file` / `--output_file` / `--model_path`.

### T5 task decision

What the sources say. The paper (arXiv:2106.14321) derives an "instruction-to-execution task", which is predicting the board state from the instructions; the old T5 target (the 180-cell board state as text) and the per-cell classifiers match that task. Pointing the other way: the earlier README (by Egecan Esen) says "T5 for instruction simplification", the old `inference.py` wrote a `simplified_instructions` column, and the notebook builds GPT-4o simplified instructions. The original code was internally inconsistent (board-state target, simplification-style inference), so either could have been meant.

Decision of this repository, not confirmed by the authors: T5 is trained here as **instruction simplification**: input `simplify instructions: <one instruction>`, target the GPT-4o simplified rewrite of that step. The `no_color` variant takes the GPT-4o colour-free instruction as input and, as target, the simplified text with colour words removed by a fixed word list (`strip_colors`, not GPT). Board-state prediction is no longer trained by `T5_Training.py`; the per-cell classifiers still cover that task. TODO(sarp): confirm with Egecan Esen which task the project intends; if it is board-state prediction, revert the T5 target.

The rule-based no-colour target can read badly. Examples (from the tests): "Make the hexagon green instead of red" becomes "Make the hexagon instead of"; "colour of the sky" is left unchanged because only listed colour words are removed. Lists ("green and yellow"), hyphenated ("Red-orange") and capitalised colours are handled, and "white" (the empty cell) is kept on purpose. Treat no-colour T5 results with that in mind.

## Tech stack

Python, PyTorch, Hugging Face Transformers, pandas, scikit-learn, pytest (tests), OpenAI API (notebook only).

## Quickstart

```bash
pip install -r requirements.txt
```

`requirements.txt` uses lower bounds set to the versions installed and checked in a Python 3.12 venv. To run the notebook, also install `jupyter` (or `ipykernel`); it is not in `requirements.txt`.

Tests (CPU, tiny synthetic data, no downloads, no network):

```bash
pip install pytest
python -m pytest tests -q
```

What was and was not run:

- Run: the tests above, `py_compile` and `pyflakes` on every script, and the BiLSTM abstraction classifier on the real Hexagons data on CPU (see Results). A one-step `t5-small` smoke run of `prepare_t5_data.py` and `T5_Training.py` is described under Results.
- `make_abstraction_xlsx.py` was run on the real Hexagons data; its output has the same instructions, levels, splits, drawing ids and steps as the file used for the numbers below.
- Never run on real data: the three `classificationbased*.py` scripts, `classification_evaluation.py`, `deberta_abs.py` (mDeBERTa on CPU is too slow here), and `inference.py`. Their training / prediction code is exercised only through a tiny randomly initialised DeBERTa in the tests. The notebook's OpenAI cells were not run.

Usage:

```bash
cd Project
# 1. run Data Preprocess.ipynb (needs the Hexagons data and `export OPENAI_API_KEY=...`); it writes df_no_color.xlsx and expanded_df_final.xlsx
python prepare_t5_data.py --input_file df_no_color.xlsx --output_file t5_data.xlsx
python T5_Training.py --data_file t5_data.xlsx --include_abstraction_level --epochs 50 --seed 42   # or --no_color; with --no_color the abstraction flag is ignored
python inference.py --device cuda --model_path path/to/model --batch_size 100 --show_time_remaining --input_file in.xlsx --output_file out.xlsx   # add --include_abstraction_level if trained with it; a --no_color model needs --instruction_column no_color
python classificationbased.py --seed 42 [--class_weighted]   # also classificationbased_nocolor.py, classificationbased-abstraction.py
python classification_evaluation.py --variant color   # or nocolor / abstraction
python deberta_abs.py --seed 42 [--class_weighted]
python lstm_abs.py --seed 42 [--class_weighted]
```

T5 arguments: `--data_file` (required), `--include_abstraction_level`, `--no_color`, `--epochs` (default 50), `--model_name`, `--batch_size`, `--max_length`, `--max_batches` (smoke tests), `--seed`. Inference arguments: `--device`, `--model_path`, `--batch_size`, `--show_time_remaining`, `--instruction_column`, `--include_abstraction_level`, `--input_file`, `--output_file`.

Class imbalance: most per-cell labels are 0, so accuracy alone misleads. `--class_weighted` (classification and abstraction scripts) uses an inverse-frequency weighted loss, off by default to keep the original behaviour; evaluation always prints macro-F1 and the majority-class baseline.

## Reproducibility notes

- Every script has `--seed` (default 42), which seeds python, numpy and torch and the training loader's shuffle. Runs on GPU can still differ because some CUDA kernels are non-deterministic.
- The OpenAI calls use `gpt-4o` and the notebook reads the key from the environment variable `OPENAI_API_KEY` (never store it in a file); GPT outputs were saved as pickles that are not committed, and several drawings were repaired by hand in the notebook (the cell that builds `new_instructions_clean`). The simplified output is merged by `merge_simplified`, which only accepts a drawing whose step count matches; mismatched drawings are listed and left empty (and dropped by `prepare_t5_data.py`), not repaired.
- Intermediate files (`df_no_color.xlsx`, `expanded_df_final.xlsx`, `model*.pth`) are not in the repo and are git-ignored. The notebook writes `df_no_color.xlsx` (and reads it back, converting `resulting_labels` from a string to a list) only since the fix in this version.

## Limitations

- Course project, not a maintained library. CI is not set up; tests are run locally only.
- The T5 task was changed to instruction simplification (see above). It has not been trained on the real GPT-4o targets here, so there is no T5 result. The GPT-4o simplified text is unreviewed, and drawings whose GPT output has the wrong step count are dropped, not repaired.
- The no-colour T5 target is rule-based: it strips listed colour words in any sense, ignores unlisted colour phrases ("the colour of the sky"), and can leave ungrammatical text (examples above).
- The GPT simplification prompt sees the whole drawing and writes standalone steps (for example "Column 9, Row 6" for "that one"), while T5 gets a single instruction, so context-dependent targets cannot be predicted from the input. The original design had this too.
- The classifier input carries no board state, only the cell position and the instruction history. This is a design weakness and was not changed.
- Class weighting is optional and off by default; per-class metrics (beyond macro-F1) are not reported.
- Changed behaviour in this version (not validated against the authors' earlier runs): the BiLSTM output now ignores padding; `classification_evaluation.py` predicts dev and test only unless `--splits` says otherwise; the T5 target and prompt changed; `T5_Training.py` loads the model inside `main` and evaluates the best checkpoint on the test split.
- Earlier fixes, also not run on real data: `classification_evaluation.py` no longer shuffles its loader; T5 labels mask padding with -100; `deberta_abs.py` moves the model to the device and uses 4 classes (also the BiLSTM); BiLSTM F1 is computed over the whole epoch.

## Credits and license

Course project for COMP 542 at Koç University by Sarp Vulaş and Egecan Esen. Dataset and task: Lachmy et al. (see above). Code licensed under Apache-2.0 (see `LICENSE`).

Author: Sarp Vulaş, Dubai, MSc Computational Finance, King's College London. LinkedIn: TODO(sarp): add LinkedIn URL.
