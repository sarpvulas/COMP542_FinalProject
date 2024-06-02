import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import argparse

def load_model_and_tokenizer(model_path, device):
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    return model, tokenizer

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

    generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

    return generated_texts

def main(args):
    # Load the model and tokenizer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)

    # Load the Excel file
    df = pd.read_excel(args.input_file)

    # Batch processing with GPU utilization and time estimation
    batch_size = args.batch_size
    total_instances = len(df)
    total_batches = (total_instances + batch_size - 1) // batch_size
    start_time = time.time()

    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, total_instances)
        batch_indices = range(batch_start, batch_end)
        instructions = df.loc[batch_indices, 'instructions'].tolist()
        input_texts = [f"simplify instructions: {instr} simplified_instructions: " for instr in instructions]

        results = generate_inference_batch(model, tokenizer, input_texts, device)
        for idx, result in zip(batch_indices, results):
            df.loc[idx, 'simplified_instructions'] = result

        if args.show_time_remaining:
            batch_end_time = time.time()
            elapsed_time = batch_end_time - start_time
            batches_completed = batch_num + 1
            average_time_per_batch = elapsed_time / batches_completed
            estimated_total_time = average_time_per_batch * total_batches
            estimated_time_remaining = estimated_total_time - elapsed_time

            print(f"Batch {batches_completed}/{total_batches} completed")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            print(f"Estimated total time: {estimated_total_time:.2f} seconds")
            print(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds")

    # Save the updated DataFrame back to an Excel file
    df.to_excel(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate simplified instructions using a T5 model.')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run the model on (cuda or cpu).')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained T5 model.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing.')
    parser.add_argument('--show_time_remaining', action='store_true', help='Show estimated time remaining for processing.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input Excel file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output Excel file.')

    args = parser.parse_args()
    main(args)
