import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import UserSequenceDataset, DualHead, build_user_chunks, DATA_CSV

CHECKPOINT_PATH = "../models/dual_head_model_seq2seq.pt"
TEST_CSV = "../data/test_labels_subtask1.csv"
OUTPUT_CSV = "predictions.csv"
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNK_SHUFFLE_SEED = 0


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]

    model = DualHead(config["model_name"], config["head_hidden_size"], config["dropout"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    return model, tokenizer, config["max_length"], config["chunk_size"], config["min_user_texts"]


def main():
    model, tokenizer, max_length, chunk_size, min_user_texts = load_model(CHECKPOINT_PATH)

    df = pd.read_csv(TEST_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Test-User, die auch im Training vorkamen, markieren ("seen" vs "unseen")
    # -- rein informativ, TRAIN-Daten fliessen hier an keiner Stelle in den
    # Kontext/die Chunks ein, nur ihre User-IDs werden zum Abgleich gelesen.
    train_user_ids = set(pd.read_csv(DATA_CSV)["user_id"].unique())
    df["seen"] = df["user_id"].isin(train_user_ids)

    chunks = build_user_chunks(df, chunk_size, min_user_texts, seed=UNK_SHUFFLE_SEED)
    loader = DataLoader(
        UserSequenceDataset(chunks, tokenizer, max_length, chunk_size),
        batch_size=BATCH_SIZE, shuffle=False
    )

    valence_by_id = {}
    arousal_by_id = {}

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            cls_positions  = batch["cls_positions"].to(DEVICE)
            valid_mask     = batch["valid_mask"]
            text_ids       = batch["text_id"]

            valence_logits, arousal_logits = model(input_ids, attention_mask, cls_positions)
            valence_logits = valence_logits.cpu()
            arousal_logits = arousal_logits.cpu()

            for b in range(text_ids.size(0)):
                for s in range(text_ids.size(1)):
                    if valid_mask[b, s] == 1:
                        tid = text_ids[b, s].item()
                        valence_by_id[tid] = valence_logits[b, s].item()
                        arousal_by_id[tid] = arousal_logits[b, s].item()

    df["valence_preds"] = df["text_id"].map(valence_by_id)
    df["arousal_preds"] = df["text_id"].map(arousal_by_id)
    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()
