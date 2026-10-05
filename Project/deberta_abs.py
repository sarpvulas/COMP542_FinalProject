import argparse

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hexagons_common import (add_seed_arg, class_weights, load_abstraction_df, majority_baseline,
                             metrics_dict, set_seed)

NUM_CLASSES = 4


# Custom dataset
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['instructions']
        label = self.dataframe.iloc[idx]['abstraction_label']
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


print_per = 100


# Training loop
def train(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    total_loss = 0
    total_batches = len(data_loader)
    for batch_idx, batch in tqdm(enumerate(data_loader)):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % print_per == 0:
            print(f"Batch {batch_idx + 1}/{total_batches}, Batch Loss: {loss.item():.4f}")
    average_loss = total_loss / total_batches
    print(f"Average Training Loss: {average_loss:.4f}")
    return average_loss


# Validation loop
def validate(model, data_loader, loss_fn, device):
    model = model.eval()
    total_loss = 0
    total_batches = len(data_loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            if (batch_idx + 1) % print_per == 0:
                print(f"Validation Batch {batch_idx + 1}/{total_batches}, Batch Loss: {loss.item():.4f}")
    average_loss = total_loss / total_batches
    print(f"Average Validation Loss: {average_loss:.4f}")
    return average_loss


def preprocess(texts, tokenizer, max_length=512):
    # Tokenize the text input for the model
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")


def validate_on_strings(model, texts, tokenizer, device):
    model = model.eval()
    data_loader = DataLoader(texts, batch_size=10)
    predictions = []
    with torch.no_grad():
        for batch_idx, texts in tqdm(enumerate(data_loader)):
            batch = preprocess(texts, tokenizer)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            predictions.extend(probs.tolist())
    return predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description="mDeBERTa abstraction-level classifier (4 groups).")
    parser.add_argument("--input_file", default="df_no_color.xlsx")
    parser.add_argument("--output_file", default="deberta_results.xlsx")
    parser.add_argument("--model_path", default="deberta_abstraction.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--class_weighted", action="store_true",
                        help="Weight the loss by inverse class frequency.")
    add_seed_arg(parser)
    args = parser.parse_args(argv)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    set_seed(args.seed)
    df = load_abstraction_df(args.input_file)
    train_df = df[df["dataset"] == "train"].copy(deep=True)
    dev_df = df[df["dataset"] == "dev"].copy(deep=True)

    model_id = "microsoft/mdeberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=NUM_CLASSES)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(TextDataset(train_df, tokenizer), batch_size=args.batch_size, shuffle=True, generator=generator)
    valid_loader = DataLoader(TextDataset(dev_df, tokenizer), batch_size=args.batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")
    model.to(device)
    weight = None
    if args.class_weighted:
        weight = class_weights(train_df["abstraction_label"].values, NUM_CLASSES).to(device)
        print(f"Class weights: {weight.tolist()}")
    loss_fn = CrossEntropyLoss(weight=weight)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = train(model, train_loader, loss_fn, optimizer, device)
        valid_loss = validate(model, valid_loader, loss_fn, device)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

    torch.save(model.state_dict(), args.model_path)

    df["bert_preds"] = validate_on_strings(model, df["instructions"].values, tokenizer, device)
    df.to_excel(args.output_file)
    df["bert_preds_final"] = [np.argmax(i) for i in df["bert_preds"]]

    train_labels = df.loc[df["dataset"] == "train", "abstraction_label"]
    rows = []
    for split, group in df.groupby("dataset", sort=False):
        m = metrics_dict(group["abstraction_label"].tolist(), group["bert_preds_final"].tolist())
        b = majority_baseline(train_labels, group["abstraction_label"].tolist())
        rows.append({"split": split, "n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "majority_accuracy": b["accuracy"], "majority_macro_f1": b["macro_f1"]})
    print(pd.DataFrame(rows).set_index("split"))


if __name__ == "__main__":
    main()
