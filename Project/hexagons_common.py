"""Shared helpers for the Hexagons scripts: seeding, the T5 prompt, abstraction labels, metrics.

Importing this module is cheap and never downloads anything.
"""
import random

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# --- T5 task: board prediction (the paper's instruction-to-execution task) ---------------
# One prompt string, used by prepare_t5_data.py (training input) and inference.py.
T5_PROMPT_PREFIX = "draw board: "


def build_t5_input(instruction_text):
    """Return the T5 input string for the instruction text (see prepare_t5_data.py)."""
    return f"{T5_PROMPT_PREFIX}{instruction_text}"


def with_abstraction_level(t5_input, level):
    """Prefix a T5 input with the abstraction level (the `--include_abstraction_level` variant)."""
    return f"Abstraction Level: {level} {t5_input}"


def model_input(t5_instr, abstraction_level=None, include_abstraction_level=False):
    """The exact string fed to T5, for training (T5_Training.py) and inference (inference.py)."""
    if include_abstraction_level:
        return with_abstraction_level(str(t5_instr), abstraction_level)
    return str(t5_instr)


# --- Abstraction labels ------------------------------------------------------------------
ABSTRACTION_LEVELS = {
    'simple': 0,
    'symmetry': 1,
    'other': 1,
    'composed objects': 2,
    'conditions': 2,
    'bounded iteration': 3,
    'conditional iteration': 3,
    'recursion': 3,
    'NONE': 2,
}


def load_abstraction_df(path):
    """Read the preprocessed Excel file and add the 4-group `abstraction_label` column."""
    df = pd.read_excel(path, index_col=0)
    df["abstraction_label"] = df["abstraction_level"].map(ABSTRACTION_LEVELS)
    return df[~df["instructions"].isna()]  # 1 instance like this


# --- Seeds -------------------------------------------------------------------------------
def set_seed(seed):
    """Seed python, numpy and torch (CPU and CUDA)."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_seed_arg(parser, default=42):
    parser.add_argument("--seed", type=int, default=default, help="Random seed (default: %(default)s).")


# --- Class imbalance and metrics ----------------------------------------------------------
def class_weights(labels, num_classes):
    """Inverse-frequency weights, normalised to mean 1 over the classes present.

    Classes absent from `labels` get weight 0 (they cannot be learned from this data).
    """
    import torch
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    weights = np.zeros(num_classes)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float)


def metrics_dict(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def majority_baseline(train_labels, y_true):
    """Metrics of always predicting the most frequent training label."""
    values, counts = np.unique(np.asarray(train_labels), return_counts=True)
    majority = values[np.argmax(counts)]
    return metrics_dict(list(y_true), [majority] * len(y_true))
