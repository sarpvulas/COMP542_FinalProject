"""Board states as text for T5, and scoring of decoded boards.

A Hexagons board has 10 rows x 18 columns = 180 cells (row-major, as in the dataset), each an
integer label 0..7 (0 = empty/white, 1..7 = the seven paint colours).

Text format (`render_board`): each row is its 18 labels separated by single spaces, and the 10
rows are joined with " / ". Example row: "0 0 3 3 0 ...". `parse_board` is the exact inverse
and rejects anything else (wrong row count, wrong row length, non-integer or out-of-range label).

No-colour board (`collapse_board`): labels 1..7 (every paint colour) become 1 ("filled"),
label 0 stays 0 ("unfilled"). This is the same collapse as the `action_label_nocolor` of the
classifiers. A cell painted white is indistinguishable from an empty cell (both are 0 in the data).
"""
ROWS = 10
COLS = 18
CELLS = ROWS * COLS
ROW_SEP = " / "


def render_board(labels):
    labels = [int(x) for x in labels]
    if len(labels) != CELLS:
        raise ValueError(f"a board has {CELLS} cells, got {len(labels)}")
    rows = [" ".join(str(x) for x in labels[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
    return ROW_SEP.join(rows)


def collapse_board(labels):
    """Colour labels 1..7 -> 1 (filled); 0 stays 0 (unfilled)."""
    return [0 if int(x) == 0 else 1 for x in labels]


def parse_board(text, max_label=7):
    """Return (board, None) for a valid text or (None, reason) for a malformed one. Never raises."""
    if not isinstance(text, str):
        return None, "not a string"
    rows = text.strip().split("/")
    if len(rows) != ROWS:
        return None, f"expected {ROWS} rows, got {len(rows)}"
    board = []
    for i, row in enumerate(rows):
        tokens = row.split()
        if len(tokens) != COLS:
            return None, f"row {i + 1}: expected {COLS} cells, got {len(tokens)}"
        for tok in tokens:
            if not (tok.isascii() and tok.isdigit()) or not 0 <= int(tok) <= max_label:
                return None, f"row {i + 1}: bad label {tok!r}"
            board.append(int(tok))
    return board, None


def score_boards(predicted_texts, gold_texts, max_label=7):
    """Score decoded boards against gold boards (both in text format).

    cell_accuracy: correct cells over all cells of all examples; a malformed prediction counts
    as every cell wrong. exact_match: share of examples whose whole board is right (per drawing
    step). blank_board_cell_accuracy: the same cell accuracy for always predicting an empty board.
    `errors` maps example position to the parse reason of each malformed prediction.
    """
    if len(predicted_texts) != len(gold_texts):
        raise ValueError("predictions and gold have different lengths")
    n = len(gold_texts)
    correct_cells = exact = blank_correct = 0
    errors = {}
    for i, (pred, gold) in enumerate(zip(predicted_texts, gold_texts)):
        gold_board, gold_error = parse_board(gold, max_label)
        if gold_board is None:
            raise ValueError(f"gold board {i} is malformed: {gold_error}")
        blank_correct += sum(1 for g in gold_board if g == 0)
        board, error = parse_board(pred, max_label)
        if board is None:
            errors[i] = error
            continue
        hits = sum(1 for p, g in zip(board, gold_board) if p == g)
        correct_cells += hits
        exact += hits == CELLS
    total = n * CELLS
    return {
        "n": n,
        "n_malformed": len(errors),
        "cell_accuracy": correct_cells / total if n else float("nan"),
        "exact_match": exact / n if n else float("nan"),
        "blank_board_cell_accuracy": blank_correct / total if n else float("nan"),
        "errors": errors,
    }
