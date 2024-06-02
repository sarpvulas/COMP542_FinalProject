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
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=8)

state_dict = torch.load('model.pth')
model.load_state_dict(state_dict)

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
        text = self.dataframe.iloc[idx]['final_input']
        label = self.dataframe.iloc[idx]['action_label']
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

# Create dataset objects
all_dataset = HexagonsDataset(df, tokenizer)

# Create dataloaders
all_loader = DataLoader(all_dataset, batch_size=25, shuffle=True)

def get_all_predictions(model, data_loader, device):
    model = model.eval()  # Set the model to evaluation mode
    predictions = []
    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)  
            
            predictions.extend(preds.cpu().numpy()) 
    
    return predictions

predictions = get_all_predictions(model, all_loader, device)

df["predictions"] = predictions
df.to_excel("predictiondf.xlsx")

