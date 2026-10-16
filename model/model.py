import torch
import torch.nn as nn

class FinancialMoodBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_categories=5, num_moods=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True, 
            dropout=0.3
        )
        # Shared Layer
        self.fc_shared = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        
        # Dual Output Heads
        self.fc_category = nn.Linear(64, num_categories)
        self.fc_mood = nn.Linear(64, num_moods)

    def forward(self, x):
        embeds = self.embedding(x)
        out, _ = self.lstm(embeds)
        
        # Global Average Pooling ข้ามทุก Token ใน Sequence
        out = torch.mean(out, dim=1)
        out = self.relu(self.fc_shared(out))
        
        cat_out = self.fc_category(out)
        mood_out = self.fc_mood(out)
        
        return cat_out, mood_out