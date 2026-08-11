"""
attention_analysis.py
Baustein 3: Attention-Masse auf den User-ID-Tokens vs. restlichem Content
(letzter Encoder-Layer, ueber alle Heads gemittelt). Grober Proxy -- fuer
praezisere Attribution waere Integrated Gradients vorzuziehen.

Ausfuehren im Ordner 2.2_UserID: python attention_analysis.py
"""
import os
import pandas as pd
import torch

from predict import load_model, CHECKPOINT_PATH, TEST_CSV, UNKNOWN_USER
from run import GROUPS

OUT_DIR = "explainability_out"
os.makedirs(OUT_DIR, exist_ok=True)
N_SAMPLES_PER_GROUP = 200


def token_group_masses(model, tokenizer, texts, effective_ids, user_id_map, max_length, uid_len):
    device = next(model.parameters()).device
    results = []

    model.eval()
    with torch.no_grad():
        for text, uid_key in zip(texts, effective_ids):
            encoding = tokenizer(text, truncation=True, padding="max_length",
                                  max_length=max_length, return_tensors="pt")
            input_ids = encoding["input_ids"][0]
            attention_mask = encoding["attention_mask"][0]

            uid_tokens = torch.tensor(user_id_map[uid_key], dtype=torch.long)
            input_ids = torch.cat([input_ids[:1], uid_tokens, input_ids[1:]])[:max_length]
            uid_mask = torch.ones(len(uid_tokens), dtype=torch.long)
            attention_mask = torch.cat([attention_mask[:1], uid_mask, attention_mask[1:]])[:max_length]

            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)

            out = model.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            attn = out.attentions[-1][0].mean(dim=0)   # [seq, seq], letzter Layer, ueber Heads gemittelt

            valid = attention_mask[0].bool()
            attn_received = (attn * valid.unsqueeze(0)).sum(dim=0)
            attn_received = attn_received / attn_received.sum().clamp(min=1e-9)

            uid_start, uid_end = 1, 1 + uid_len   # Position 0 = CLS, direkt danach der User-ID-Block
            mass_uid = attn_received[uid_start:uid_end].sum().item()
            mass_content = 1.0 - mass_uid

            results.append({"mass_uid_tokens": mass_uid, "mass_content": mass_content})

    return pd.DataFrame(results)


def main():
    model, tokenizer, max_length, user_id_map, user_mapping = load_model(CHECKPOINT_PATH)
    uid_len = len(next(iter(user_id_map.values())))

    df = pd.read_csv(TEST_CSV)
    user_ids = df["user_id"].tolist()
    df["effective_id"] = [user_mapping.get(uid, UNKNOWN_USER) for uid in user_ids]

    all_rows = []
    for name, mask_fn in GROUPS:
        sub = df[mask_fn(df)]
        if len(sub) == 0:
            continue
        sample = sub.sample(n=min(N_SAMPLES_PER_GROUP, len(sub)), random_state=42)

        masses = token_group_masses(
            model, tokenizer, sample["text"].tolist(),
            sample["effective_id"].tolist(), user_id_map, max_length, uid_len,
        )
        masses["group"] = name
        all_rows.append(masses)
        print(f"{name:<14} n={len(sample):>4}  "
              f"mass_uid={masses['mass_uid_tokens'].mean():.4f}  "
              f"mass_content={masses['mass_content'].mean():.4f}")

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(os.path.join(OUT_DIR, "attention_mass_by_group.csv"), index=False)
    result.groupby("group")[["mass_uid_tokens", "mass_content"]].mean() \
        .to_csv(os.path.join(OUT_DIR, "attention_mass_summary.csv"))


if __name__ == "__main__":
    main()
