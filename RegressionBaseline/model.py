import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel

class AffectDataset(Dataset):
    def __init__(self, texts, valence_labels, arousal_labels, 
                 tokenizer, max_length):
        self.texts = texts
        self.valence_labels = valence_labels
        self.arousal_labels = arousal_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "valence": torch.tensor(self.valence_labels[idx], dtype=torch.float),
            "arousal": torch.tensor(self.arousal_labels[idx], dtype=torch.float),
        }

class RegressionHead(nn.Module):
    def __init__(self, input_dim, hidden_size=None, dropout=0.1):
        super().__init__()
        if hidden_size is None:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DualHead(nn.Module):
    def __init__(self, model_name, head_hidden_size, dropout, pooling_strategy):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.valence_head = RegressionHead(hidden_size, head_hidden_size, dropout)
        self.arousal_head = RegressionHead(hidden_size, head_hidden_size, dropout)

        self.pooling_strategy = pooling_strategy
        self.dropout = nn.Dropout(dropout)

    def _pool(self, encoder_output, attention_mask):
        if self.pooling_strategy == 'cls':
            return encoder_output.last_hidden_state[:, 0, :]

        if self.pooling_strategy == 'mean':
            mask = attention_mask.unsqueeze(-1).float()
            summed = (encoder_output.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return summed / counts

        raise ValueError(f"Unbekannte pooling_strategy: {self.pooling_strategy}")

    def forward(self, input_ids, attention_mask):
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(self._pool(encoder_output, attention_mask))
        return self.valence_head(pooled), self.arousal_head(pooled)


# Konfiguration
MODEL_NAME        = "bert-base-uncased"
MAX_LENGTH        = 128
BATCH_SIZE        = 16
DROPOUT           = 0.1
POOLING_STRATEGY  = "cls"
NUM_EPOCHS        = 5
LEARNING_RATE     = 2e-5
HEAD_HIDDEN_SIZE  = 256
DATA_CSV          = "../data/train_subtask1.csv"
VAL_SPLIT         = 0.2
SEED              = 42
SAVE_PATH         = "../models/dual_head_model_regression.pt"
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import pandas as pd
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    texts = df['text'].tolist()

    valence = df['valence'].astype(float).tolist()
    arousal = df['arousal'].astype(float).tolist()
    return texts, valence, arousal


def run_epoch(model, loader, optimizer, scheduler, criterion, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            valence_labels = batch['valence'].to(DEVICE)
            arousal_labels = batch['arousal'].to(DEVICE)

            valence_logits, arousal_logits = model(input_ids, attention_mask)
            loss = criterion(valence_logits, valence_labels) + criterion(arousal_logits, arousal_labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
    return total_loss / len(loader)

import random

def main():
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    texts, valence, arousal = load_data(DATA_CSV)

    train_texts, val_texts, train_val, val_val, train_aro, val_aro = \
        train_test_split(texts, 
                         valence, 
                         arousal, 
                         test_size=VAL_SPLIT, 
                         random_state=SEED)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        AffectDataset(train_texts, train_val, train_aro,
                      tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        AffectDataset(val_texts, val_val, val_aro, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = DualHead(
        MODEL_NAME, HEAD_HIDDEN_SIZE, DROPOUT, POOLING_STRATEGY
    ).to(DEVICE)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps  = len(train_loader) * NUM_EPOCHS
    scheduler    = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion    = nn.MSELoss()
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, criterion, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, scheduler, criterion, train=False)
        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'model_name': MODEL_NAME,
                    'max_length': MAX_LENGTH,
                    'head_hidden_size': HEAD_HIDDEN_SIZE,
                    'dropout': DROPOUT,
                    'pooling_strategy': POOLING_STRATEGY,
    },
}, SAVE_PATH)

if __name__ == "__main__":
    main()