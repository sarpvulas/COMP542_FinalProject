import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from hexagons_common import (add_seed_arg, class_weights, load_abstraction_df, majority_baseline,
                             metrics_dict, set_seed)

NUM_CLASSES = 4


class HexagonsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['instructions']
        labels = self.data.iloc[idx]['abstraction_label']
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, attention_mask=None):
        """`attention_mask` (1 = real token, right-padded) lets the LSTM skip padding, so the
        final hidden states are those of the last real token (forward) and the first (backward)."""
        embedded = self.dropout(self.embedding(text))
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
            embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        lstm_output, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over `loader`; trains if an optimizer is given. Returns (loss sum, preds, labels)."""
    model.train(optimizer is not None)
    total_loss, preds, labels_all = 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            predictions = model(input_ids, mask)
            loss = loss_fn(predictions, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            preds.extend(predictions.argmax(dim=1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    return total_loss, preds, labels_all


def predict_probs(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            out.extend(torch.softmax(model(batch['input_ids'].to(device), batch['attention_mask'].to(device)), dim=-1).tolist())
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="BiLSTM abstraction-level classifier (4 groups).")
    parser.add_argument("--input_file", default="df_no_color.xlsx")
    parser.add_argument("--output_file", default="lstm_results.xlsx")
    parser.add_argument("--model_path", default="lstm_model.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--class_weighted", action="store_true",
                        help="Weight the loss by inverse class frequency.")
    add_seed_arg(parser)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    set_seed(args.seed)
    df = load_abstraction_df(args.input_file)
    train_df = df[df["dataset"] == "train"].copy(deep=True)
    dev_df = df[df["dataset"] == "dev"].copy(deep=True)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(HexagonsDataset(train_df, tokenizer), batch_size=args.batch_size, shuffle=True, generator=generator)
    dev_loader = DataLoader(HexagonsDataset(dev_df, tokenizer), batch_size=args.batch_size, shuffle=False)
    all_loader = DataLoader(HexagonsDataset(df, tokenizer), batch_size=args.batch_size, shuffle=False)

    model = BiLSTM(vocab_size=tokenizer.vocab_size, embedding_dim=256, hidden_dim=128, output_dim=NUM_CLASSES, num_layers=2, bidirectional=True, dropout=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    weight = None
    if args.class_weighted:
        weight = class_weights(train_df["abstraction_label"].values, NUM_CLASSES).to(device)
        print(f"Class weights: {weight.tolist()}")
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    for epoch in range(args.epochs):
        loss, preds, labels = run_epoch(model, train_loader, loss_fn, device, optimizer)
        m = metrics_dict(labels, preds)
        print(f"Epoch {epoch+1}, Training Loss: {loss:.4f}, Accuracy: {m['accuracy']:.4f}, F1 Score: {m['macro_f1']:.4f}")
        loss, preds, labels = run_epoch(model, dev_loader, loss_fn, device)
        m = metrics_dict(labels, preds)
        print(f"Validation Loss: {loss:.4f}, Accuracy: {m['accuracy']:.4f}, F1 Score: {m['macro_f1']:.4f}")

    torch.save(model.state_dict(), args.model_path)

    df["lstm_predictions"] = predict_probs(model, all_loader, device)
    df.to_excel(args.output_file)
    df["lstm_preds_final"] = [np.argmax(i) for i in df["lstm_predictions"]]

    train_labels = df.loc[df["dataset"] == "train", "abstraction_label"]
    rows = []
    for split, group in df.groupby("dataset", sort=False):
        m = metrics_dict(group["abstraction_label"].tolist(), group["lstm_preds_final"].tolist())
        b = majority_baseline(train_labels, group["abstraction_label"].tolist())
        rows.append({"split": split, "n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "majority_accuracy": b["accuracy"], "majority_macro_f1": b["macro_f1"]})
    print(pd.DataFrame(rows).set_index("split"))


if __name__ == "__main__":
    main()
