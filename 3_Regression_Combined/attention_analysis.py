"""
attention_analysis.py
Baustein 3: Attention-Masse auf User-ID- und Datums-Tokens vs. restlichem
Content (letzter Encoder-Layer, ueber alle Heads gemittelt).

Zusaetzlich zur rohen Masse wird ein laengennormalisierter Enrichment-Wert
berichtet: das Verhaeltnis der beobachteten Masse zu der Masse, die die
Tokens unter uniformer Attention allein aufgrund ihrer Anzahl erhalten
wuerden (n_tokens / seq_len). Ein Wert von 1.0 bedeutet "genau wie durch
Zufall zu erwarten", Werte > 1 bedeuten ueberproportionale Beachtung.

Ausfuehren im Ordner 3_Regression_Combined: python attention_analysis.py
"""
import os
import pandas as pd
import torch

from model import format as date_format
from predict import load_model, CHECKPOINT_PATH, TEST_CSV, UNKNOWN_USER
from run import GROUPS

OUT_DIR = "explainability_out"
os.makedirs(OUT_DIR, exist_ok=True)
N_SAMPLES_PER_GROUP = None  # None = alle Texte der Gruppe verwenden

COLUMNS = [
    "mass_uid_tokens", "mass_date_tokens", "mass_cls", "mass_content",
    "enrichment_uid", "enrichment_date", "enrichment_content",
    "n_uid_tokens", "n_date_tokens", "seq_len",
]


def token_group_masses(model, tokenizer, texts, time_strs, effective_ids, user_id_map, max_length, uid_len):
    device = next(model.parameters()).device
    prefix_len_cache = {}
    results = []

    model.eval()
    with torch.no_grad():
        for text, time_str, uid_key in zip(texts, time_strs, effective_ids):
            encoding = tokenizer(text, truncation=True, padding="max_length",
                                  max_length=max_length, return_tensors="pt")
            input_ids = encoding["input_ids"][0]
            attention_mask = encoding["attention_mask"][0]

            uid_tokens = torch.tensor(user_id_map[uid_key], dtype=torch.long)
            input_ids = torch.cat([input_ids[:1], uid_tokens, input_ids[1:]])[:max_length]
            uid_mask = torch.ones(len(uid_tokens), dtype=torch.long)
            attention_mask = torch.cat([attention_mask[:1], uid_mask, attention_mask[1:]])[:max_length]

            if time_str not in prefix_len_cache:
                prefix_len_cache[time_str] = len(tokenizer(time_str, add_special_tokens=False)["input_ids"])
            p_len = prefix_len_cache[time_str]

            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)

            out = model.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            attn = out.attentions[-1][0].mean(dim=0)   # [seq, seq], letzter Layer, ueber Heads gemittelt

            valid = attention_mask[0].bool()
            attn_received = (attn * valid.unsqueeze(1)).sum(dim=0)
            attn_received = attn_received / attn_received.sum().clamp(min=1e-9)

            seq_len = int(valid.sum().item())
            uid_start, uid_end = 1, min(1 + uid_len, seq_len)   # Position 0 = CLS
            date_start, date_end = uid_end, min(uid_end + p_len, seq_len)
            n_uid = max(uid_end - uid_start, 0)
            n_date = max(date_end - date_start, 0)
            n_content = seq_len - n_uid - n_date - 1

            mass_uid = attn_received[uid_start:uid_end].sum().item()
            mass_date = attn_received[date_start:date_end].sum().item()
            mass_cls = attn_received[0].item()
            mass_content = max(1.0 - mass_uid - mass_date - mass_cls, 0.0)

            exp_uid = n_uid / seq_len if seq_len > 0 else float("nan")
            exp_date = n_date / seq_len if seq_len > 0 else float("nan")
            exp_content = n_content / seq_len if seq_len > 0 else float("nan")

            results.append({
                "mass_uid_tokens": mass_uid,
                "mass_date_tokens": mass_date,
                "mass_cls": mass_cls,
                "mass_content": mass_content,
                "enrichment_uid": mass_uid / exp_uid if exp_uid and exp_uid > 0 else float("nan"),
                "enrichment_date": mass_date / exp_date if exp_date and exp_date > 0 else float("nan"),
                "enrichment_content": mass_content / exp_content if exp_content and exp_content > 0 else float("nan"),
                "n_uid_tokens": n_uid,
                "n_date_tokens": n_date,
                "seq_len": seq_len,
            })

    return pd.DataFrame(results)


def main():
    model, tokenizer, max_length, user_id_map, user_mapping = load_model(CHECKPOINT_PATH)
    uid_len = len(next(iter(user_id_map.values())))

    df = pd.read_csv(TEST_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["time_str"] = df["timestamp"].dt.strftime(date_format)
    df["full_text"] = df["time_str"] + " " + df["text"]

    user_ids = df["user_id"].tolist()
    df["effective_id"] = [user_mapping.get(uid, UNKNOWN_USER) for uid in user_ids]

    all_rows = []
    for name, mask_fn in GROUPS:
        sub = df[mask_fn(df)]
        if len(sub) == 0:
            continue
        if N_SAMPLES_PER_GROUP is not None and len(sub) > N_SAMPLES_PER_GROUP:
            sub = sub.sample(n=N_SAMPLES_PER_GROUP, random_state=42)

        masses = token_group_masses(
            model, tokenizer,
            sub["full_text"].tolist(), sub["time_str"].tolist(),
            sub["effective_id"].tolist(), user_id_map, max_length, uid_len,
        )
        masses["group"] = name
        all_rows.append(masses)
        print(f"{name:<14} n={len(sub):>5}  "
              f"mass_uid={masses['mass_uid_tokens'].mean():.4f} "
              f"(enrich {masses['enrichment_uid'].mean():.3f})  "
              f"mass_date={masses['mass_date_tokens'].mean():.4f} "
              f"(enrich {masses['enrichment_date'].mean():.3f})  "
              f"mass_cls={masses['mass_cls'].mean():.4f}  "
              f"mass_content={masses['mass_content'].mean():.4f} "
              f"(enrich {masses['enrichment_content'].mean():.3f})  "
              f"mean_seq_len={masses['seq_len'].mean():.1f}")

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(os.path.join(OUT_DIR, "attention_mass_by_group.csv"), index=False)
    result.groupby("group")[COLUMNS].mean() \
        .to_csv(os.path.join(OUT_DIR, "attention_mass_summary.csv"))
    print(f"\nErgebnisse in {OUT_DIR}/ gespeichert.")


if __name__ == "__main__":
    main()
