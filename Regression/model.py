import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel

class AffectDataset(Dataset):
    def __init__(self, texts, valence_labels, arousal_labels, 
                 tokenizer, max_length, user_ids=None, uid_map=None):
        self.texts = texts
        self.valence_labels = valence_labels
        self.arousal_labels = arousal_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.user_ids = user_ids
        self.uid_map = uid_map

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

        if self.uid_map is not None and self.user_ids is not None:
            uid_tokens = self.uid_map[self.user_ids[idx]]
            uid_tensor = torch.tensor(uid_tokens, dtype=torch.long)
            input_ids = torch.cat([input_ids[:1], uid_tensor, input_ids[1:]])[:self.max_length]
            uid_mask = torch.ones(len(uid_tokens), dtype=torch.long)
            attention_mask = torch.cat([attention_mask[:1], uid_mask, attention_mask[1:]])[:self.max_length]

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
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
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
MODEL_NAME        = "cardiffnlp/twitter-roberta-base-emotion"
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
SAVE_PATH         = "../models/dual_head_model.pt"
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#configuration for user ID handling
MIN_USER_TEXTS = 15
UNKNOWN_USER = "UNKNOWN"
USER_ID_LENGTH = 3      #L in paper


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

format = 'year: %Y month: %m day: %d'

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    df['timestamp'] = pd.to_datetime(df['collection_phase'])
    df['time_str'] = df['timestamp'].dt.strftime(format)
    
    texts = (df['timestamp'] + " " + df['text']).tolist()

    valence = df['valence'].astype(float).tolist()
    arousal = df['arousal'].astype(float).tolist()
    user_ids = df['user_id'].to_list()
    return texts, valence, arousal, user_ids


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

def build_user_mapping(user_ids, min_texts=MIN_USER_TEXTS):
    from collections import Counter
    counts = Counter(user_ids)

    mapping = {}
    for user_id in counts:
        if counts[user_id] < min_texts:
            mapping[user_id] = UNKNOWN_USER
        else:
            mapping[user_id] = user_id

    return mapping

#function to generate token-sequences for user IDs
import random

def generate_user_identifiers(user_mapping, tokenizer, L=USER_ID_LENGTH, seed=SEED):
    random.seed(seed)
    vocab_size = tokenizer.vocab_size

    effective_ids = set(user_mapping.values())
    effective_ids.add(UNKNOWN_USER)

    user_id_map = {}
    for eid in effective_ids:
        user_id_map[eid] = random.sample(range(vocab_size), L)

    return user_id_map

def main():
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    texts, valence, arousal, user_ids = load_data(DATA_CSV)

    user_mapping = build_user_mapping(user_ids, MIN_USER_TEXTS)

    effective_ids = [user_mapping[uid] for uid in user_ids]

    user_id_map = generate_user_identifiers(user_mapping, tokenizer)

    train_texts, val_texts, train_val, val_val, train_aro, val_aro, train_uids, val_uids = \
        train_test_split(texts, 
                         valence, 
                         arousal, 
                         effective_ids,
                         test_size=VAL_SPLIT, 
                         random_state=SEED)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        AffectDataset(train_texts, train_val, train_aro,
                      tokenizer, MAX_LENGTH, train_uids, user_id_map),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        AffectDataset(val_texts, val_val, val_aro, tokenizer, MAX_LENGTH, val_uids, user_id_map),
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
                'user_id_map': user_id_map,       # <- neu
                'user_mapping': user_mapping,     # <- neu
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