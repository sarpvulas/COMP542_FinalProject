import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim, nn
from sklearn.metrics import accuracy_score, f1_score


df = pd.read_excel("df_no_color.xlsx", index_col=0)

abstraction_levels = {
    'simple': 0,
    'symmetry': 1,
    'other': 1,
    'composed objects': 2,
    'conditions': 2,
    'bounded iteration': 3,
    'conditional iteration': 3,
    'recursion': 3,
    'NONE': 2
}

df["abstraction_label"] = df["abstraction_level"].map(abstraction_levels)
df = df[~df["instructions"].isna()] # 1 instance like this

train_df = df[df["dataset"] == "train"].copy(deep=True)
dev_df = df[df["dataset"] == "dev"].copy(deep=True)


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


# Tokenizer and DataLoader
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
train_dataset = HexagonsDataset(train_df, tokenizer)
dev_dataset = HexagonsDataset(dev_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=256, shuffle=False)

all_dataset = HexagonsDataset(df, tokenizer)
all_loader = DataLoader(all_dataset, batch_size=256, shuffle=False)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        lstm_output, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)


# Initialize model
model = BiLSTM(vocab_size=tokenizer.vocab_size, embedding_dim=256, hidden_dim=128, output_dim=4, num_layers=2, bidirectional=True, dropout=0.5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()


model_save_path = "lstm_model"

def calculate_metrics(all_preds, all_labels):
    # Computed once per epoch over every prediction, not averaged per batch.
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return accuracy, f1


# Training and Validation Loop
for epoch in range(20):
    model.train()
    total_train_loss = 0
    train_preds, train_labels = [], []

    for batch in train_loader:
        # Move batch data to the device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        predictions = model(input_ids)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_preds.extend(predictions.argmax(dim=1).cpu().tolist())
        train_labels.extend(labels.cpu().tolist())

    avg_train_accuracy, avg_train_f1 = calculate_metrics(train_preds, train_labels)
    print(f'Epoch {epoch+1}, Training Loss: {total_train_loss:.4f}, Accuracy: {avg_train_accuracy:.4f}, F1 Score: {avg_train_f1:.4f}')

    # Validation after each epoch
    model.eval()
    total_val_loss = 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for batch in dev_loader:
            # Move batch data to the device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            predictions = model(input_ids)
            loss = loss_fn(predictions, labels)
            total_val_loss += loss.item()
            val_preds.extend(predictions.argmax(dim=1).cpu().tolist())
            val_labels.extend(labels.cpu().tolist())

    avg_val_accuracy, avg_val_f1 = calculate_metrics(val_preds, val_labels)
    print(f'Validation Loss: {total_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}, F1 Score: {avg_val_f1:.4f}')

torch.save(model.state_dict(), f"{model_save_path}.pth")

def validate_on_strings(model, all_loader, tokenizer, device):
    predictions = []
    accuracies = []
    f1_scores = []

    with torch.no_grad():
        for batch in all_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            probs = torch.softmax(outputs, dim=-1)
            predictions.extend(probs.tolist())

    return predictions


df["lstm_predictions"] = validate_on_strings(model, all_loader, tokenizer, device)

df.to_excel("lstm_results.xlsx")

df["lstm_preds_final"] = [np.argmax(i) for i in df["lstm_predictions"]]

def calculate_metrics(group):
    accuracy = accuracy_score(group['abstraction_label'], group['lstm_preds_final'])
    f1 = f1_score(group['abstraction_label'], group['lstm_preds_final'], average='macro')
    return pd.Series({'Accuracy': accuracy, 'F1 Score': f1})

# Calculate metrics for each dataset group
results = df.groupby('dataset').apply(calculate_metrics)

print(results)
