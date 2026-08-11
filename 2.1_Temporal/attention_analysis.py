"""
attention_analysis.py
Baustein 3: Attention-Masse auf den Datums-Prefix-Tokens vs. restlichem
Content (letzter Encoder-Layer, ueber alle Heads gemittelt). Grober Proxy --
fuer praezisere Attribution waere Integrated Gradients vorzuziehen.

Ausfuehren im Ordner 2.1_Temporal: python attention_analysis.py
"""
import os
import pandas as pd
import torch

from predict import load_model, load_test_data, CHECKPOINT_PATH, TEST_CSV
from run import GROUPS

OUT_DIR = "explainability_out"
os.makedirs(OUT_DIR, exist_ok=True)
N_SAMPLES_PER_GROUP = 200


def token_group_masses(model, tokenizer, texts, date_strs, max_length):
    device = next(model.parameters()).device
    prefix_len_cache = {}
    results = []

    model.eval()
    with torch.no_grad():
        for text, date_str in zip(texts, date_strs):
            encoding = tokenizer(text, truncation=True, padding="max_length",
                                  max_length=max_length, return_tensors="pt")
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            if date_str not in prefix_len_cache:
                prefix_len_cache[date_str] = len(tokenizer(date_str, add_special_tokens=False)["input_ids"])
            p_len = prefix_len_cache[date_str]

            out = model.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            attn = out.attentions[-1][0].mean(dim=0)   # [seq, seq], letzter Layer, ueber Heads gemittelt

            valid = attention_mask[0].bool()
            attn_received = (attn * valid.unsqueeze(0)).sum(dim=0)
            attn_received = attn_received / attn_received.sum().clamp(min=1e-9)

            seq_len = int(valid.sum().item())
            date_start, date_end = 1, 1 + p_len  # Position 0 = CLS, direkt danach der Datums-Prefix

            mass_date = attn_received[date_start:min(date_end, seq_len)].sum().item()
            mass_content = 1.0 - mass_date

            results.append({"mass_date_tokens": mass_date, "mass_content": mass_content})

    return pd.DataFrame(results)


def main():
    model, tokenizer, max_length = load_model(CHECKPOINT_PATH)

    _, _, _, df = load_test_data(TEST_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date_str"] = df["timestamp"].dt.strftime("year: %Y month: %m day: %d")
    df["full_text"] = df["date_str"] + " " + df["text"]

    all_rows = []
    for name, mask_fn in GROUPS:
        sub = df[mask_fn(df)]
        if len(sub) == 0:
            continue
        sample = sub.sample(n=min(N_SAMPLES_PER_GROUP, len(sub)), random_state=42)

        masses = token_group_masses(
            model, tokenizer, sample["full_text"].tolist(), sample["date_str"].tolist(), max_length,
        )
        masses["group"] = name
        all_rows.append(masses)
        print(f"{name:<14} n={len(sample):>4}  "
              f"mass_date={masses['mass_date_tokens'].mean():.4f}  "
              f"mass_content={masses['mass_content'].mean():.4f}")

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(os.path.join(OUT_DIR, "attention_mass_by_group.csv"), index=False)
    result.groupby("group")[["mass_date_tokens", "mass_content"]].mean() \
        .to_csv(os.path.join(OUT_DIR, "attention_mass_summary.csv"))


if __name__ == "__main__":
    main()
