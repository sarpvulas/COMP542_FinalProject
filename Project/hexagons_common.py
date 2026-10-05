"""Shared helpers for the Hexagons scripts: seeding, the T5 prompt, GPT merge, metrics.

Importing this module is cheap and never downloads anything.
"""
import random
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# --- T5 task: instruction simplification -------------------------------------------------
# One prompt string, used by prepare_t5_data.py (training input) and inference.py.
T5_PROMPT_PREFIX = "simplify instructions: "


def build_t5_input(instruction):
    """Return the exact T5 input string for one instruction."""
    return f"{T5_PROMPT_PREFIX}{instruction}"


def with_abstraction_level(t5_input, level):
    """Prefix a T5 input with the abstraction level (the `--include_abstraction_level` variant)."""
    return f"Abstraction Level: {level} {t5_input}"


# --- GPT-4o output handling --------------------------------------------------------------
# The notebook joins the steps of a drawing with this marker before sending them to GPT-4o.
STEP_MARKER = "[END OF CURRENT INSTRUCTION, NEXT INSTRUCTION NOW]"
_STEP_SPLIT = re.compile(r"\s*" + re.escape(STEP_MARKER) + r"\s*")

# Digits 0..7 of the board state are white, black, yellow, green, red, blue, purple, orange
# (Hexagons README). White is the empty cell, so it is not stripped as a "colour word".
# The seven paint colours plus common synonyms found in free-text instructions. "white" is the
# empty cell and is kept on purpose ("color white hexagons black" -> "color white hexagons").
COLOR_WORDS = ("black", "yellow", "green", "red", "blue", "purple", "orange", "pink", "brown",
               "gray", "grey", "navy", "violet", "cyan", "magenta", "teal", "turquoise", "maroon",
               "gold", "golden", "lime", "indigo", "lavender", "beige")
_COLOR = r"(?:" + "|".join(COLOR_WORDS) + r")"
# A colour, hyphenated compounds ("red-orange"), and lists of colours joined by commas / and / or / & / slash.
_COLOR_TERM = _COLOR + r"(?:-" + _COLOR + r")*"
_COLOR_RUN_RE = re.compile(
    r"(?<!\w)" + _COLOR_TERM + r"(?:\s*(?:,|/|&)\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+)(?=" + _COLOR_TERM + r")",
    re.IGNORECASE)
_COLOR_RE = re.compile(r"(?<!\w)" + _COLOR_TERM + r"(?!\w)", re.IGNORECASE)
_FILLER_SEGMENTS = {"", "then", "and", "or", "but", "also"}


_NUMBERING = re.compile(r"^\s*(?:step\s+)?\d+\s*[.)]\s+", re.IGNORECASE)


def _strip_numbering(step):
    """Drop a leading "1. " / "2) " / "Step 3. " (the GPT prompt forbids numbering but models add it)."""
    return _NUMBERING.sub("", str(step), count=1).strip()


def split_steps(text):
    """Split one GPT output (all steps of a drawing) into per-step strings."""
    return [part.strip() for part in _STEP_SPLIT.split(str(text).strip())]


def merge_simplified(df, simplified_by_drawing):
    """Add a `simplified_instructions` column to `df` from GPT-4o output.

    `simplified_by_drawing` maps `id_of_drawing` to the raw GPT text (steps joined by
    STEP_MARKER) or to a ready list of step strings. A drawing is merged only when the
    number of returned steps equals its number of rows; otherwise its rows get NaN and its
    id is returned in the second value, so nothing is silently misaligned.
    Rows are matched by order of `step_number` within a drawing.
    """
    if not df.index.is_unique:
        raise ValueError("merge_simplified needs a unique DataFrame index")
    if df.duplicated(["id_of_drawing", "step_number"]).any():
        raise ValueError("duplicated (id_of_drawing, step_number) pairs; steps cannot be matched by order")
    out = df.copy()
    out["simplified_instructions"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    mismatched = []
    for drawing_id, group in out.groupby("id_of_drawing", sort=False):
        raw = simplified_by_drawing.get(drawing_id)
        if raw is None:
            mismatched.append(drawing_id)
            continue
        steps = [_strip_numbering(x) for x in (raw if isinstance(raw, (list, tuple)) else split_steps(raw))]
        ordered = group.sort_values("step_number")
        if len(steps) != len(ordered):
            mismatched.append(drawing_id)
            continue
        out.loc[ordered.index, "simplified_instructions"] = steps
    return out, mismatched


def strip_colors(text):
    """Remove colour words (see COLOR_WORDS) from a string; used for the no-colour T5 target.

    Colour lists ("green and yellow"), hyphenated compounds ("Red-orange") and capitalised
    colours are removed whole; clauses left empty or holding only "then/and/or/but/also" are dropped.
    The rule is not grammatical in general (see README): "instead of red" leaves "instead of".
    Text without a colour word is returned unchanged (apart from outer whitespace).
    """
    text = str(text)
    if not _COLOR_RE.search(text):
        return text.strip()
    text = _COLOR_RUN_RE.sub("", text)
    text = _COLOR_RE.sub("", text)
    parts = re.split(r"([;,.:])", text)
    kept = []
    for i in range(0, len(parts), 2):
        segment = re.sub(r"\s+", " ", parts[i]).strip()
        delimiter = parts[i + 1] if i + 1 < len(parts) else ""
        if segment.lower() in _FILLER_SEGMENTS:
            continue
        kept.append(segment + delimiter)
    return re.sub(r"[;,:]+$", "", " ".join(kept).strip()).strip()


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
