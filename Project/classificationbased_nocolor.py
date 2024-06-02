import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn import DataParallel
from tqdm import tqdm


# Read the prepared df
df = pd.read_excel("expanded_df_final.xlsx",index_col=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for training")

model_id = "microsoft/mdeberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

if torch.cuda.device_count() > 1:
    model = DataParallel(model)
model.to(device)

# Custom dataset
class HexagonsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['final_input_nocolor']
        label = self.dataframe.iloc[idx]['action_label_nocolor']
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

train_df = df[df['dataset'] == 'train']
valid_df = df[df['dataset'] == 'dev']

# Create dataset objects
train_dataset = HexagonsDataset(train_df, tokenizer)
valid_dataset = HexagonsDataset(valid_df, tokenizer)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=25, shuffle=False)

optimizer = Adam(model.parameters(), lr=3e-5)
loss_fn = CrossEntropyLoss()


print_per = 1000

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

# Actual training of model
for epoch in range(20):  
    print(f"Epoch {epoch + 1}")
    train_loss = train(model, train_loader, loss_fn, optimizer, device)
    valid_loss = validate(model, valid_loader, loss_fn, device)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
    torch.save(model.module.state_dict(), 'model_nocolor.pth')


