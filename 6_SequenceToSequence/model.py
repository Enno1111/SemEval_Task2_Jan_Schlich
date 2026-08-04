import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel

CHUNK_SIZE = 5          # Texte pro User-Sequenz
MAX_LENGTH = 512        # Gesamtlänge der Sequenz (mehrere Texte + Marker)
MIN_USER_TEXTS = CHUNK_SIZE   # User mit weniger Texten landen im gemeinsamen UNK-Pool


class UserSequenceDataset(Dataset):
    """
    Ein Sample = eine Sequenz aus bis zu CHUNK_SIZE Texten (desselben Users,
    oder aus dem UNK-Pool), als [CLS] text [SEP] [CLS] text [SEP] ...
    aneinandergereiht (BERTSUM-Stil). Die Vorhersage fuer Text i wird am
    Hidden State der i-ten [CLS]-Position abgegriffen.
    """
    def __init__(self, chunks, tokenizer, max_length, chunk_size):
        self.chunks = chunks   # Liste von Chunks; Chunk = Liste von (text, valence, arousal, text_id)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size

        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        self.per_text_budget = max(1, max_length // chunk_size - 2)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]

        input_ids = []
        cls_positions = []
        valid_mask = []
        valence_labels = []
        arousal_labels = []
        text_ids = []

        for slot in range(self.chunk_size):
            if slot < len(chunk):
                text, valence, arousal, text_id = chunk[slot]

                body_ids = self.tokenizer(
                    text, truncation=True, max_length=self.per_text_budget,
                    add_special_tokens=False,
                )["input_ids"]

                cls_positions.append(len(input_ids))
                input_ids.append(self.cls_id)
                input_ids.extend(body_ids)
                input_ids.append(self.sep_id)

                valid_mask.append(1)
                valence_labels.append(valence)
                arousal_labels.append(arousal)
                text_ids.append(text_id)
            else:
                cls_positions.append(0)
                valid_mask.append(0)
                valence_labels.append(0.0)
                arousal_labels.append(0.0)
                text_ids.append(-1)

        input_ids = input_ids[:self.max_length]
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len

        for i, pos in enumerate(cls_positions):
            if valid_mask[i] == 1 and pos >= self.max_length:
                valid_mask[i] = 0
                cls_positions[i] = 0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "cls_positions": torch.tensor(cls_positions, dtype=torch.long),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.float),
            "valence": torch.tensor(valence_labels, dtype=torch.float),
            "arousal": torch.tensor(arousal_labels, dtype=torch.float),
            "text_id": torch.tensor(text_ids, dtype=torch.long),
        }


import random


def build_user_chunks(df, chunk_size, min_user_texts=1, seed=0):
    """
    User mit >= min_user_texts Texten bekommen eigene, chronologisch
    sortierte Chunks. User darunter landen gemeinsam in einem gemischten
    UNK-Pool, der genauso in chunk_size-Gruppen aufgeteilt wird (kein
    personalisierter Kontext, aber auch kein verschwendetes Padding).
    """
    counts = df.groupby("user_id").size()
    known_users = counts[counts >= min_user_texts].index
    unk_users = counts[counts < min_user_texts].index

    chunks = []

    for user_id in known_users:
        group = df[df["user_id"] == user_id].sort_values("timestamp")
        rows = list(zip(group["text"], group["valence"].astype(float),
                         group["arousal"].astype(float), group["text_id"]))
        for i in range(0, len(rows), chunk_size):
            chunks.append(rows[i:i + chunk_size])

    if len(unk_users) > 0:
        unk_df = df[df["user_id"].isin(unk_users)]
        unk_rows = list(zip(unk_df["text"], unk_df["valence"].astype(float),
                             unk_df["arousal"].astype(float), unk_df["text_id"]))
        rng = random.Random(seed)
        rng.shuffle(unk_rows)
        for i in range(0, len(unk_rows), chunk_size):
            chunks.append(unk_rows[i:i + chunk_size])

    return chunks


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
    def __init__(self, model_name, head_hidden_size, dropout):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.gradient_checkpointing_enable()
        hidden_size = self.encoder.config.hidden_size

        self.valence_head = RegressionHead(hidden_size, head_hidden_size, dropout)
        self.arousal_head = RegressionHead(hidden_size, head_hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, cls_positions):
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_output.last_hidden_state   # (batch, seq_len, hidden)

        batch_size, chunk_size = cls_positions.shape
        index = cls_positions.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        slot_hidden = torch.gather(hidden_states, 1, index)   # (batch, chunk_size, hidden)
        slot_hidden = self.dropout(slot_hidden)

        flat = slot_hidden.reshape(batch_size * chunk_size, -1)
        valence = self.valence_head(flat).reshape(batch_size, chunk_size)
        arousal = self.arousal_head(flat).reshape(batch_size, chunk_size)
        return valence, arousal


# Konfiguration
MODEL_NAME        = "microsoft/deberta-base-mnli"
BATCH_SIZE        = 4
DROPOUT           = 0.1
NUM_EPOCHS        = 5
LEARNING_RATE     = 2e-5
HEAD_HIDDEN_SIZE  = None
DATA_CSV          = "../data/train_subtask1.csv"
VAL_SPLIT         = 0.2
SEED              = 42
SAVE_PATH         = "../models/dual_head_model_seq2seq.pt"
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import pandas as pd
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
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def run_epoch(model, loader, optimizer, scheduler, criterion, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            cls_positions  = batch["cls_positions"].to(DEVICE)
            valid_mask     = batch["valid_mask"].to(DEVICE)
            valence_labels = batch["valence"].to(DEVICE)
            arousal_labels = batch["arousal"].to(DEVICE)

            valence_logits, arousal_logits = model(input_ids, attention_mask, cls_positions)

            valence_loss = (criterion(valence_logits, valence_labels) * valid_mask).sum() / valid_mask.sum()
            arousal_loss = (criterion(arousal_logits, arousal_labels) * valid_mask).sum() / valid_mask.sum()
            loss = valence_loss + arousal_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = load_data(DATA_CSV)

    user_ids = df["user_id"].unique().tolist()
    train_users, val_users = train_test_split(user_ids, test_size=VAL_SPLIT, random_state=SEED)

    train_df = df[df["user_id"].isin(train_users)]
    val_df   = df[df["user_id"].isin(val_users)]

    train_chunks = build_user_chunks(train_df, CHUNK_SIZE, MIN_USER_TEXTS, seed=SEED)
    val_chunks   = build_user_chunks(val_df,   CHUNK_SIZE, MIN_USER_TEXTS, seed=SEED)

    train_loader = DataLoader(
        UserSequenceDataset(train_chunks, tokenizer, MAX_LENGTH, CHUNK_SIZE),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        UserSequenceDataset(val_chunks, tokenizer, MAX_LENGTH, CHUNK_SIZE),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model = DualHead(MODEL_NAME, HEAD_HIDDEN_SIZE, DROPOUT).to(DEVICE)

    optimizer     = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps   = len(train_loader) * NUM_EPOCHS
    scheduler     = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion     = nn.MSELoss(reduction='none')
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
                    'head_hidden_size': HEAD_HIDDEN_SIZE,
                    'dropout': DROPOUT,
                    'max_length': MAX_LENGTH,
                    'chunk_size': CHUNK_SIZE,
                    'min_user_texts': MIN_USER_TEXTS,
                },
            }, SAVE_PATH)

if __name__ == "__main__":
    main()
