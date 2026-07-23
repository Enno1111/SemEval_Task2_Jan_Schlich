import pandas as pd
import torch
from transformers import AutoTokenizer
from model import AffectDataset, DualHead

CHECKPOINT_PATH = "../models/dual_head_model.pt"
TEST_CSV = "../data/test_labels_subtask1.csv"
OUTPUT_CSV = "predictions.csv"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNKNOWN_USER = "UNKNOWN"

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    user_id_map = checkpoint['user_id_map']
    user_mapping = checkpoint['user_mapping']

    model = DualHead(
        config["model_name"],
        config["head_hidden_size"],
        config["dropout"],
        config["pooling_strategy"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    return model, tokenizer, config["max_length"], user_id_map, user_mapping

def load_test_data(csv_path):
    df = pd.read_csv(csv_path)

    texts = df['text'].tolist()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    from model import format
    df['time_str'] = df['timestamp'].dt.strftime(format)

    texts = (df['time_str'] + " " + df['text']).tolist()
    
    user_ids = df['user_id'].tolist()
    dummy_labels = [0] * len(texts)
    return texts, dummy_labels, dummy_labels, user_ids, df

from torch.utils.data import DataLoader

def predict(model, loader):
    valence_preds, arousal_preds = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            valence_logits, arousal_logits = model(input_ids, attention_mask)

            valence_preds.extend(valence_logits.cpu().tolist())
            arousal_preds.extend(arousal_logits.cpu().tolist())

    return valence_preds, arousal_preds

def main():
    model, tokenizer, max_length, user_id_map, user_mapping = load_model(CHECKPOINT_PATH)
    texts, dummy_valence, dummy_arousal, user_ids, df = load_test_data(TEST_CSV)

    effective_ids = [user_mapping.get(uid, UNKNOWN_USER) for uid in user_ids]

    test_loader = DataLoader(
        AffectDataset(texts, dummy_valence, dummy_arousal,
                      tokenizer, max_length, effective_ids, user_id_map),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    valence_preds, arousal_preds = predict(model, test_loader)

    df['valence_preds'] = valence_preds
    df['arousal_preds'] = arousal_preds
    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()