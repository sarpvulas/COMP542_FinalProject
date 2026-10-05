"""Shared code for the per-cell action classifiers (mDeBERTa) and their evaluation.

`classificationbased.py`, `classificationbased_nocolor.py` and `classificationbased-abstraction.py`
are thin wrappers that call `train_main(variant)`; `classification_evaluation.py` calls
`evaluate_main()`. Nothing here runs at import time and nothing is downloaded until main runs.
"""
import argparse
import json

import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hexagons_common import (add_seed_arg, class_weights, majority_baseline, metrics_dict,
                             set_seed)

MODEL_ID = "microsoft/mdeberta-v3-base"

VARIANTS = {
    "color": dict(text_column="final_input", label_column="action_label", num_labels=8,
                  model_path="model.pth"),
    "nocolor": dict(text_column="final_input_nocolor", label_column="action_label_nocolor",
                    num_labels=2, model_path="model_nocolor.pth"),
    "abstraction": dict(text_column="abstraction_input", label_column="action_label", num_labels=8,
                        model_path="model_abstraction.pth"),
}


class HexagonsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, text_column, label_column, max_len=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx][self.text_column]
        label = self.dataframe.iloc[idx][self.label_column]
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


def save_state_dict(model, path):
    """Save weights whether or not the model is wrapped in DataParallel."""
    torch.save(getattr(model, "module", model).state_dict(), path)


def train_epoch(model, data_loader, loss_fn, optimizer, device, print_per=1000):
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


def validate_epoch(model, data_loader, loss_fn, device, print_per=1000):
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


def fit(model, tokenizer, df, variant, device, epochs=20, batch_size=25, lr=3e-5,
        class_weighted=False, model_path=None, seed=42):
    """Train on df[dataset == 'train'], validate on 'dev', save weights after every epoch."""
    cfg = VARIANTS[variant]
    model_path = model_path or cfg["model_path"]
    train_df = df[df['dataset'] == 'train']
    valid_df = df[df['dataset'] == 'dev']
    make = lambda d: HexagonsDataset(d, tokenizer, cfg["text_column"], cfg["label_column"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(make(train_df), batch_size=batch_size, shuffle=True, generator=generator)
    valid_loader = DataLoader(make(valid_df), batch_size=batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=lr)
    weight = None
    if class_weighted:
        weight = class_weights(train_df[cfg["label_column"]].values, cfg["num_labels"]).to(device)
        print(f"Class weights: {weight.tolist()}")
    loss_fn = CrossEntropyLoss(weight=weight)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        valid_loss = validate_epoch(model, valid_loader, loss_fn, device)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
        save_state_dict(model, model_path)


def predict(model, data_loader, device):
    """Argmax predictions in loader order (the loader must not shuffle)."""
    model = model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask=attention_mask).logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return predictions


def split_report(df, label_column, pred_column="predictions"):
    """Accuracy and macro-F1 per split, with the majority-class baseline of the train labels.

    `df` must hold every split whose baseline is wanted, plus the train rows for the
    majority class (they need no predictions: only `label_column` is read for them).
    """
    train_labels = df.loc[df['dataset'] == 'train', label_column]
    report = {}
    for split, group in df.groupby('dataset', sort=False):
        if pred_column not in group or group[pred_column].isna().all():
            continue
        report[split] = {
            "model": metrics_dict(group[label_column].tolist(), group[pred_column].tolist()),
            "majority_baseline": majority_baseline(train_labels, group[label_column].tolist()),
        }
    return report


def print_report(report):
    for split, res in report.items():
        m, b = res["model"], res["majority_baseline"]
        print(f"{split:>5} n={m['n']}  accuracy={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}"
              f"  | majority baseline accuracy={b['accuracy']:.4f} macro_f1={b['macro_f1']:.4f}")


def _add_common_args(parser, variant):
    cfg = VARIANTS[variant]
    parser.add_argument("--input_file", default="expanded_df_final.xlsx",
                        help="Excel file written by Data Preprocess.ipynb (default: %(default)s).")
    parser.add_argument("--model_path", default=cfg["model_path"],
                        help="Weights file (default: %(default)s).")
    add_seed_arg(parser)


def train_main(variant, argv=None):
    parser = argparse.ArgumentParser(description=f"Train the '{variant}' action classifier ({MODEL_ID}).")
    _add_common_args(parser, variant)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--class_weighted", action="store_true",
                        help="Weight the loss by inverse class frequency (most cell labels are 0).")
    args = parser.parse_args(argv)

    from torch.nn import DataParallel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(args.seed)
    df = pd.read_excel(args.input_file, index_col=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=VARIANTS[variant]["num_labels"])
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(device)
    fit(model, tokenizer, df, variant, device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        class_weighted=args.class_weighted, model_path=args.model_path, seed=args.seed)


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description="Predict with a trained action classifier and report "
                                                 "accuracy and macro-F1 per split.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="color")
    parser.add_argument("--input_file", default="expanded_df_final.xlsx")
    parser.add_argument("--model_path", default=None, help="Weights file (default: the variant's default).")
    parser.add_argument("--output_file", default="predictiondf.xlsx", help="Predictions Excel file.")
    parser.add_argument("--metrics_file", default=None, help="JSON metrics file (default: metrics_<variant>.json).")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["train", "dev", "test"],
                        help="Splits to predict and report (default: dev test).")
    parser.add_argument("--batch_size", type=int, default=25)
    add_seed_arg(parser)
    args = parser.parse_args(argv)
    cfg = VARIANTS[args.variant]
    model_path = args.model_path or cfg["model_path"]
    metrics_file = args.metrics_file or f"metrics_{args.variant}.json"

    from torch.nn import DataParallel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(args.seed)
    df = pd.read_excel(args.input_file, index_col=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=cfg["num_labels"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(device)

    selected = df[df['dataset'].isin(args.splits)].copy()
    loader = DataLoader(HexagonsDataset(selected, tokenizer, cfg["text_column"], cfg["label_column"]),
                        batch_size=args.batch_size, shuffle=False)  # shuffle=False: predictions are assigned by position
    selected["predictions"] = predict(model, loader, device)
    selected.to_excel(args.output_file)

    # Train rows are only needed for the majority class of the baseline.
    with_train = pd.concat([df[df['dataset'] == 'train'], selected]) if 'train' not in args.splits else selected
    report = split_report(with_train, cfg["label_column"])
    print_report(report)
    with open(metrics_file, "w") as f:
        json.dump({"variant": args.variant, "model_path": model_path, "seed": args.seed, "splits": report}, f, indent=2)
    print(f"Metrics saved to {metrics_file}; predictions to {args.output_file}")
