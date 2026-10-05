import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import time
import argparse

from hexagons_common import add_seed_arg, model_input, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HexagonsDataset(Dataset):
    def __init__(self, data_file, dataset_type, tokenizer, max_length=512, include_abstraction_level=False,
                 no_color=False):
        """
        Initialize the dataset. Loads data and sets tokenizer and other parameters.

        Args:
            data_file (str): Path to the dataset file.
            dataset_type (str): Type of dataset (train, dev, test).
            tokenizer (T5Tokenizer): Tokenizer to encode text.
            max_length (int, optional): Max length for encoding. Defaults to 512.
            include_abstraction_level (bool, optional): Include abstraction level in input. Defaults to False.
            no_color (bool, optional): Use dataset without color info. Defaults to False.
        """
        self.data = self.load_data(data_file, dataset_type, no_color)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_abstraction_level = include_abstraction_level
        self.no_color = no_color

    def load_data(self, data_file, dataset_type, no_color):
        """
        Load data from an Excel file and filter based on dataset type and color info.

        Args:
            data_file (str): Path to the dataset file.
            dataset_type (str): Type of dataset (train, dev, test).
            no_color (bool): Use dataset without color info.

        Returns:
            pd.DataFrame: Filtered dataset.
        """
        df = pd.read_excel(data_file, index_col=0)
        if no_color:
            dataset = df[df['dataset'] == dataset_type][['t5_instr_no_color', 'resulting_label_list_no_color']]
        else:
            dataset = df[df['dataset'] == dataset_type][['t5_instr', 'resulting_label_list', 'abstraction_level']]
        return dataset.reset_index(drop=True)

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get an item from the dataset at a given index.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: Encoded input and label tensors.
        """
        item = self.data.iloc[idx]
        if self.no_color:
            instruction = str(item['t5_instr_no_color'])
            label = str(item['resulting_label_list_no_color'])
        else:
            instruction = str(item['t5_instr'])
            label = str(item['resulting_label_list'])
            instruction = model_input(instruction, item['abstraction_level'], self.include_abstraction_level)

        input_encoding = self.tokenizer(
            instruction,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        label_encoding = self.tokenizer(
            label,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = label_encoding['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding in the loss

        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': labels
        }


def create_dataloader(file_path, dataset_type, tokenizer, batch_size=4, include_abstraction_level=False,
                      no_color=False, shuffle=False, max_length=512):
    """
    Create a DataLoader for the given dataset.

    Args:
        file_path (str): Path to the dataset file.
        dataset_type (str): Type of dataset (train, dev, test).
        tokenizer (T5Tokenizer): Tokenizer to encode text.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 4.
        include_abstraction_level (bool, optional): Include abstraction level in input. Defaults to False.
        no_color (bool, optional): Use dataset without color info. Defaults to False.
        shuffle (bool, optional): Shuffle (train only). Defaults to False.
        max_length (int, optional): Max token length of input and target. Defaults to 512.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = HexagonsDataset(file_path, dataset_type, tokenizer, max_length=max_length,
                              include_abstraction_level=include_abstraction_level, no_color=no_color)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, dataloader, optimizer, scheduler, device, max_batches=None):
    """Train the model for one epoch (at most `max_batches` batches if given). Returns the mean loss."""
    model.train()
    total_loss = 0
    n = 0
    for batch in dataloader:
        if max_batches is not None and n >= max_batches:
            break
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        n += 1
    return total_loss / max(n, 1)


def evaluate(model, dataloader, device, max_batches=None):
    """Mean loss over a dataloader (at most `max_batches` batches if given), no gradient."""
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            if max_batches is not None and n >= max_batches:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            n += 1
    return total_loss / max(n, 1)


def main(args):
    """
    Main training loop for the T5 model.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    set_seed(args.seed)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
    print(device)

    common = dict(include_abstraction_level=args.include_abstraction_level, no_color=args.no_color,
                  batch_size=args.batch_size, max_length=args.max_length)
    train_dataloader = create_dataloader(args.data_file, 'train', tokenizer, shuffle=True, **common)
    val_dataloader = create_dataloader(args.data_file, 'dev', tokenizer, **common)
    test_dataloader = create_dataloader(args.data_file, 'test', tokenizer, **common)

    optimizer = AdamW(model.parameters(), lr=3e-5)
    num_training_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device, args.max_batches)
        val_loss = evaluate(model, val_dataloader, device, args.max_batches)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Duration: {epoch_duration:.2f} seconds")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = checkpoint_name = f't5_model_checkpoint_{epoch + 1}_{"with_abstraction" if args.include_abstraction_level else "no_abstraction"}_{"no_color" if args.no_color else "with_color"}'
            model.save_pretrained(checkpoint_name)
            tokenizer.save_pretrained(checkpoint_name)
            print(f"Model and tokenizer saved at epoch {epoch + 1} with validation loss {val_loss}")

    final_model_name = f't5_final_model_{"with_abstraction" if args.include_abstraction_level else "no_abstraction"}_{"no_color" if args.no_color else "with_color"}'
    model.save_pretrained(final_model_name)
    tokenizer.save_pretrained(final_model_name)
    print("Model training complete and saved.")

    # One evaluation on the test split, with the best-dev-loss checkpoint (behaviour change:
    # the test loader used to be built and never used).
    if best_val_loss < float('inf'):
        best_model = T5ForConditionalGeneration.from_pretrained(best_checkpoint).to(device)
        test_loss = evaluate(best_model, test_dataloader, device, args.max_batches)
        print(f"Test loss ({best_checkpoint}): {test_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train T5 model on Hexagons dataset with optional abstraction levels and no color information.')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the Excel file written by prepare_t5_data.py.')
    parser.add_argument('--include_abstraction_level', action='store_true',
                        help='Include abstraction levels in the input.')
    parser.add_argument('--no_color', action='store_true', help='Use no color information dataset.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--model_name', type=str, default='google/t5-v1_1-base', help='Hugging Face model id.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
    parser.add_argument('--max_length', type=int, default=512, help='Max tokens for input and target.')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Stop each train/eval pass after this many batches (smoke tests).')
    add_seed_arg(parser)

    args = parser.parse_args()
    main(args)
