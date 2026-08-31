"""
attention_analysis.py
Baustein 3: Attention-Masse auf den Datums-Prefix-Tokens vs. restlichem
Content (letzter Encoder-Layer, ueber alle Heads gemittelt).

Zusaetzlich zur rohen Masse wird ein laengennormalisierter Enrichment-Wert
berichtet: das Verhaeltnis der beobachteten Masse zu der Masse, die die
Tokens unter uniformer Attention allein aufgrund ihrer Anzahl erhalten
wuerden (n_tokens / seq_len). Ein Wert von 1.0 bedeutet "genau wie durch
Zufall zu erwarten", Werte > 1 bedeuten ueberproportionale Beachtung. Damit
sind kurze (feeling words) und lange (essays) Texte direkt vergleichbar.

Ausfuehren im Ordner 2.1_Temporal: python attention_analysis.py
"""
import os
import pandas as pd
import torch

from predict import load_model, load_test_data, CHECKPOINT_PATH, TEST_CSV
from run import GROUPS

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
N_SAMPLES_PER_GROUP = None  # None = alle Texte der Gruppe verwenden

COLUMNS = [
    "mass_date_tokens", "mass_cls", "mass_content",
    "enrichment_date", "enrichment_content",
    "n_date_tokens", "seq_len",
]


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
            attn_received = (attn * valid.unsqueeze(1)).sum(dim=0)
            attn_received = attn_received / attn_received.sum().clamp(min=1e-9)

            seq_len = int(valid.sum().item())
            date_start, date_end = 1, min(1 + p_len, seq_len)   # Position 0 = CLS
            n_date = max(date_end - date_start, 0)
            n_content = seq_len - n_date - 1                    # ohne CLS, ohne Datums-Prefix

            mass_date = attn_received[date_start:date_end].sum().item()
            mass_cls = attn_received[0].item()
            mass_content = max(1.0 - mass_date - mass_cls, 0.0)

            exp_date = n_date / seq_len if seq_len > 0 else float("nan")
            exp_content = n_content / seq_len if seq_len > 0 else float("nan")

            results.append({
                "mass_date_tokens": mass_date,
                "mass_cls": mass_cls,
                "mass_content": mass_content,
                "enrichment_date": mass_date / exp_date if exp_date and exp_date > 0 else float("nan"),
                "enrichment_content": mass_content / exp_content if exp_content and exp_content > 0 else float("nan"),
                "n_date_tokens": n_date,
                "seq_len": seq_len,
            })

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
        if N_SAMPLES_PER_GROUP is not None and len(sub) > N_SAMPLES_PER_GROUP:
            sub = sub.sample(n=N_SAMPLES_PER_GROUP, random_state=42)

        masses = token_group_masses(
            model, tokenizer, sub["full_text"].tolist(), sub["date_str"].tolist(), max_length,
        )
        masses["group"] = name
        all_rows.append(masses)
        print(f"{name:<14} n={len(sub):>5}  "
              f"mass_date={masses['mass_date_tokens'].mean():.4f}  "
              f"enrich_date={masses['enrichment_date'].mean():.3f}  "
              f"mass_cls={masses['mass_cls'].mean():.4f}  "
              f"mass_content={masses['mass_content'].mean():.4f}  "
              f"enrich_content={masses['enrichment_content'].mean():.3f}  "
              f"mean_seq_len={masses['seq_len'].mean():.1f}")

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(os.path.join(OUT_DIR, "attention_mass_by_group.csv"), index=False)
    result.groupby("group")[COLUMNS].mean() \
        .to_csv(os.path.join(OUT_DIR, "attention_mass_summary.csv"))
    print(f"\nErgebnisse in {OUT_DIR}/ gespeichert.")


if __name__ == "__main__":
    main()
