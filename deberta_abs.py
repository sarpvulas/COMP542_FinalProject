import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, Adam
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_scheduler
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


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

model_id = "microsoft/mdeberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=5)

# Create dataset objects
train_dataset = TextDataset(train_df, tokenizer)
valid_dataset = TextDataset(valid_df, tokenizer)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

optimizer = Adam(model.parameters(), lr=4e-5)
loss_fn = CrossEntropyLoss()

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for training")

model_save_path = "deberta_abstraction"

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
            if (batch_idx + 1) % print_per == 0:  # print every 10 batches
                print(f"Validation Batch {batch_idx + 1}/{total_batches}, Batch Loss: {loss.item():.4f}")
    average_loss = total_loss / total_batches
    print(f"Average Validation Loss: {average_loss:.4f}")
    return average_loss

for epoch in range(20):
    print(f"Epoch {epoch + 1}")
    train_loss = train(model, train_loader, loss_fn, optimizer, device)
    valid_loss = validate(model, valid_loader, loss_fn, device)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

torch.save(model.state_dict(), f"{model_save_path}.pth")


def preprocess(texts, tokenizer, max_length=512):
    # Tokenize the text input for the model
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

def validate_on_strings(model, texts, tokenizer, device):
    model = model.eval()
    total_loss = 0
    data_loader = DataLoader(texts, batch_size=10)
    predictions = []
    with torch.no_grad():
        for batch_idx, texts in tqdm(enumerate(data_loader)):
            batch = preprocess(texts, tokenizer)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions.extend(probs.tolist())
    return predictions


texts = df["instructions"].values
predictions = validate_on_strings(model, texts, tokenizer, device)

df["bert_preds"] = predictions

df.to_excel("deberta_results.xlsx")

df["bert_preds_final"] = [np.argmax(i) for i in df["bert_preds"]]

def calculate_metrics(group):
    accuracy = accuracy_score(group['abstraction_label'], group['bert_preds_final'])
    f1 = f1_score(group['abstraction_label'], group['bert_preds_final'], average='macro')
    return pd.Series({'Accuracy': accuracy, 'F1 Score': f1})

# Calculate metrics for each dataset group
results = df.groupby('dataset').apply(calculate_metrics)

print(results)

