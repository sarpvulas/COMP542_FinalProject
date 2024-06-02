import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import time
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base").to(device)


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
            if self.include_abstraction_level:
                instruction = f"Abstraction Level: {item['abstraction_level']} {instruction}"

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

        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': label_encoding['input_ids'].squeeze(0)
        }


def create_dataloader(file_path, dataset_type, tokenizer, batch_size=4, include_abstraction_level=False,
                      no_color=False):
    """
    Create a DataLoader for the given dataset.

    Args:
        file_path (str): Path to the dataset file.
        dataset_type (str): Type of dataset (train, dev, test).
        tokenizer (T5Tokenizer): Tokenizer to encode text.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 4.
        include_abstraction_level (bool, optional): Include abstraction level in input. Defaults to False.
        no_color (bool, optional): Use dataset without color info. Defaults to False.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = HexagonsDataset(file_path, dataset_type, tokenizer, include_abstraction_level=include_abstraction_level,
                              no_color=no_color)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main(args):
    """
    Main training loop for the T5 model.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    train_dataloader = create_dataloader(args.data_file, 'train', tokenizer,
                                         include_abstraction_level=args.include_abstraction_level,
                                         no_color=args.no_color)
    val_dataloader = create_dataloader(args.data_file, 'dev', tokenizer,
                                       include_abstraction_level=args.include_abstraction_level, no_color=args.no_color)
    test_dataloader = create_dataloader(args.data_file, 'test', tokenizer,
                                        include_abstraction_level=args.include_abstraction_level,
                                        no_color=args.no_color)

    optimizer = AdamW(model.parameters(), lr=3e-5)
    num_training_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    def train_epoch(model, dataloader, optimizer, scheduler, device):
        """
        Train the model for one epoch.

        Args:
            model (T5ForConditionalGeneration): Model to train.
            dataloader (DataLoader): DataLoader for training data.
            optimizer (AdamW): Optimizer.
            scheduler (get_linear_schedule_with_warmup): Learning rate scheduler.
            device (torch.device): Device to train on (CPU or GPU).

        Returns:
            float: Average training loss for the epoch.
        """
        model.train()
        total_loss = 0
        for batch in dataloader:
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
        return total_loss / len(dataloader)

    def evaluate(model, dataloader, device):
        """
        Evaluate the model on validation data.

        Args:
            model (T5ForConditionalGeneration): Model to evaluate.
            dataloader (DataLoader): DataLoader for validation data.
            device (torch.device): Device to evaluate on (CPU or GPU).

        Returns:
            float: Average validation loss.
        """
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
        return total_loss / len(dataloader)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        val_loss = evaluate(model, val_dataloader, device)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Duration: {epoch_duration:.2f} seconds")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_name = f't5_model_checkpoint_{epoch + 1}_{"with_abstraction" if args.include_abstraction_level else "no_abstraction"}_{"no_color" if args.no_color else "with_color"}'
            model.save_pretrained(checkpoint_name)
            tokenizer.save_pretrained(checkpoint_name)
            print(f"Model and tokenizer saved at epoch {epoch + 1} with validation loss {val_loss}")

    final_model_name = f't5_final_model_{"with_abstraction" if args.include_abstraction_level else "no_abstraction"}_{"no_color" if args.no_color else "with_color"}'
    model.save_pretrained(final_model_name)
    tokenizer.save_pretrained(final_model_name)
    print("Model training complete and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train T5 model on Hexagons dataset with optional abstraction levels and no color information.')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the dataset file.')
    parser.add_argument('--include_abstraction_level', action='store_true',
                        help='Include abstraction levels in the input.')
    parser.add_argument('--no_color', action='store_true', help='Use no color information dataset.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')

    args = parser.parse_args()
    main(args)
