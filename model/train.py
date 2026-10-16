import os
import re
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
from pythainlp.tokenize import word_tokenize
from model import FinancialMoodBiLSTM

# 1. กำหนด Path อ้างอิงแบบ Absolute
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MODEL_DIR)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'financial_mood_bilingual.csv')

# 2. โหลดข้อมูล Dataset สองภาษา
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"ไม่พบไฟล์ Dataset ที่ {DATA_PATH} กรุณารัน `uv run python data/generate_dataset.py` ก่อน")

df = pd.read_csv(DATA_PATH)

# สร้าง Label Mapping
categories = sorted(df['category'].unique().tolist())
moods = sorted(df['trigger_mood'].unique().tolist())

cat2idx = {c: i for i, c in enumerate(categories)}
mood2idx = {m: i for i, m in enumerate(moods)}

label_encoders = {
    'categories': categories,
    'moods': moods,
    'cat2idx': cat2idx,
    'mood2idx': mood2idx
}

# บันทึก label_encoders.json ลงในโฟลเดอร์ model/
label_encoders_path = os.path.join(MODEL_DIR, 'label_encoders.json')
with open(label_encoders_path, 'w', encoding='utf-8') as f:
    json.dump(label_encoders, f, ensure_ascii=False, indent=2)

# 3. Hybrid Tokenizer (TH/EN)
def tokenize_bilingual(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9ก-๙\s]', '', text)
    if re.search(r'[ก-๙]', text):
        tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
    else:
        tokens = text.split()
    return tokens

# สร้าง Vocabulary
words = []
for text in df['text']:
    words.extend(tokenize_bilingual(text))

word_counts = Counter(words)
vocab = {'<PAD>': 0, '<UNK>': 1}
for word, _ in word_counts.most_common(5000):
    vocab[word] = len(vocab)

# บันทึก vocab.json ลงในโฟลเดอร์ model/
vocab_path = os.path.join(MODEL_DIR, 'vocab.json')
with open(vocab_path, 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

def text_to_ids(text, max_len=50):
    tokens = tokenize_bilingual(text)
    ids = [vocab.get(w, vocab['<UNK>']) for w in tokens]
    if len(ids) < max_len:
        ids += [vocab['<PAD>']] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

class FinancialDataset(Dataset):
    def __init__(self, df):
        self.inputs = [text_to_ids(t) for t in df['text']]
        self.cat_labels = [cat2idx[c] for c in df['category']]
        self.mood_labels = [mood2idx[m] for m in df['trigger_mood']]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx]),
            torch.tensor(self.cat_labels[idx]),
            torch.tensor(self.mood_labels[idx])
        )

# 4. เตรียม DataLoader & Model
dataset = FinancialDataset(df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FinancialMoodBiLSTM(
    vocab_size=len(vocab),
    num_categories=len(categories),
    num_moods=len(moods)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Loop ฝึกสอนโมเดล
print("Starting training Bilingual Multi-Output BiLSTM...")
epochs = 12
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, cat_batch, mood_batch in dataloader:
        x_batch = x_batch.to(device)
        cat_batch = cat_batch.to(device)
        mood_batch = mood_batch.to(device)
        
        optimizer.zero_grad()
        cat_preds, mood_preds = model(x_batch)
        
        loss_cat = criterion(cat_preds, cat_batch)
        loss_mood = criterion(mood_preds, mood_batch)
        combined_loss = loss_cat + loss_mood
        
        combined_loss.backward()
        optimizer.step()
        total_loss += combined_loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} - Combined Loss: {total_loss/len(dataloader):.4f}")

# บันทึก weights.pth ลงในโฟลเดอร์ model/
weights_path = os.path.join(MODEL_DIR, 'weights.pth')
torch.save(model.state_dict(), weights_path)
print(f"Model training completed & saved successfully to {weights_path}!")