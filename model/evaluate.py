import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
import re
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model import SentimentBiLSTM

# 1. โหลด Vocab และ Model
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentimentBiLSTM(vocab_size=len(vocab)).to(device)
model.load_state_dict(torch.load('weights.pth', map_location=device))
model.eval()

# 2. โหลด Test Dataset (เอาส่วนที่ไม่ซ้ำกับ 10,000 แถวแรกที่เคย Train)
df = pd.read_csv('../data/IMDB Dataset.csv')
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
test_df = df.iloc[10000:12000] # ดึงมา 2,000 แถวสำหรับวัดผล

def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().split()

def text_to_pipeline(text, max_len=100):
    tokens = clean_text(text)
    ids = [vocab.get(w, vocab['<UNK>']) for w in tokens]
    if len(ids) < max_len:
        ids += [vocab['<PAD>']] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

class IMDBDataset(Dataset):
    def __init__(self, reviews, labels):
        self.inputs = [text_to_pipeline(r) for r in reviews]
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx])

test_dataset = IMDBDataset(test_df['review'].values, test_df['label'].values)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. รันการทำนาย
all_preds = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.numpy())

# 4. แสดงผลวัดประสิทธิภาพ
print("=== Model Evaluation Results ===")
print(f"Accuracy: {accuracy_score(all_labels, all_preds) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=['Negative', 'Positive']))
print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))