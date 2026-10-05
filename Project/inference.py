import argparse
import json
import time

import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from board_text import parse_board, score_boards
from hexagons_common import add_seed_arg, model_input, set_seed


def load_model_and_tokenizer(model_path, device):
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    return model, tokenizer


def build_inputs(df, instruction_column, include_abstraction_level=False):
    """Model inputs for every row, built exactly like the training inputs (hexagons_common.model_input)."""
    levels = df['abstraction_level'] if include_abstraction_level else [None] * len(df)
    return [model_input(text, level, include_abstraction_level)
            for text, level in zip(df[instruction_column], levels)]


def generate_inference_batch(model, tokenizer, instructions, device, max_length=512):
    model.eval()
    input_encodings = tokenizer(
        instructions,
        padding='longest',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = input_encodings['input_ids'].to(device)
    attention_mask = input_encodings['attention_mask'].to(device)

    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)

    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]


def score_dataframe(df, pred_column, gold_column, max_label):
    """Score decoded board texts; malformed outputs are counted and listed, never raised."""
    result = score_boards(df[pred_column].tolist(), df[gold_column].tolist(), max_label)
    result["errors"] = {int(df.index[i]): reason for i, reason in result["errors"].items()}
    return result


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)

    instruction_column = 't5_instr_no_color' if args.no_color else 't5_instr'
    gold_column = 'resulting_label_list_no_color' if args.no_color else 'resulting_label_list'
    max_label = 1 if args.no_color else 7
    df = pd.read_excel(args.input_file, index_col=0)
    if args.split:
        df = df[df['dataset'] == args.split]
    if args.max_rows:
        df = df.head(args.max_rows)
    df = df.copy()
    # The abstraction flag only applies to the colour model, as in training.
    inputs = build_inputs(df, instruction_column, args.include_abstraction_level and not args.no_color)

    batch_size = args.batch_size
    total_batches = (len(inputs) + batch_size - 1) // batch_size
    start_time = time.time()
    predictions = []
    for batch_num in range(total_batches):
        batch = inputs[batch_num * batch_size:(batch_num + 1) * batch_size]
        predictions.extend(generate_inference_batch(model, tokenizer, batch, device, args.max_length))

        if args.show_time_remaining:
            elapsed_time = time.time() - start_time
            batches_completed = batch_num + 1
            estimated_total_time = elapsed_time / batches_completed * total_batches
            print(f"Batch {batches_completed}/{total_batches} completed")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            print(f"Estimated total time: {estimated_total_time:.2f} seconds")
            print(f"Estimated time remaining: {estimated_total_time - elapsed_time:.2f} seconds")

    df['predicted_board_text'] = predictions
    df['prediction_error'] = [parse_board(p, max_label)[1] or "" for p in predictions]
    df.to_excel(args.output_file)
    print(f"Results saved to {args.output_file}")

    if gold_column in df.columns:
        result = score_dataframe(df, 'predicted_board_text', gold_column, max_label)
        print(f"n={result['n']} malformed={result['n_malformed']} "
              f"cell_accuracy={result['cell_accuracy']:.4f} exact_match={result['exact_match']:.4f} "
              f"(blank-board cell accuracy {result['blank_board_cell_accuracy']:.4f})")
        if args.metrics_file:
            with open(args.metrics_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Metrics saved to {args.metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict boards with a trained T5 model and score them.')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run the model on (cuda or cpu).')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned T5 model.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing.')
    parser.add_argument('--max_length', type=int, default=512, help='Max tokens for input and output.')
    parser.add_argument('--show_time_remaining', action='store_true', help='Show estimated time remaining for processing.')
    parser.add_argument('--no_color', action='store_true',
                        help='Use the no-colour columns (t5_instr_no_color, resulting_label_list_no_color).')
    parser.add_argument('--include_abstraction_level', action='store_true',
                        help='Prefix the abstraction level (use for a model trained with the same flag).')
    parser.add_argument('--split', type=str, default=None, choices=['train', 'dev', 'test'],
                        help='Only predict this split (default: all rows).')
    parser.add_argument('--max_rows', type=int, default=None, help='Only the first N rows (smoke tests).')
    parser.add_argument('--input_file', type=str, required=True, help='Excel file written by prepare_t5_data.py.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output Excel file.')
    parser.add_argument('--metrics_file', type=str, default=None, help='Optional JSON file for the scores.')
    add_seed_arg(parser)

    args = parser.parse_args()
    main(args)
