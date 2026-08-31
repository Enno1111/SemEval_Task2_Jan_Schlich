"""
head_analysis.py
Baustein 4: Attention-Enrichment auf den User-ID-Tokens, aufgeschluesselt
nach Layer und Head statt nur ueber den letzten Layer gemittelt.

Motivation: Die Mittelung ueber alle Heads eines einzelnen Layers kann eine
Spezialisierung einzelner Heads verdecken. Diese Auswertung berechnet die
laengennormalisierte Enrichment fuer jede (Layer, Head)-Kombination und
aggregiert anschliessend nach Evaluationsgruppe.

Jeder Text wird genau einmal durch den Encoder geschickt; die Aggregation
nach Gruppen erfolgt danach ueber die Masken aus run.py.

Ausfuehren im Ordner 2.2_UserID: python head_analysis.py
"""
import os

import numpy as np
import pandas as pd
import torch

from predict import load_model, CHECKPOINT_PATH, TEST_CSV, UNKNOWN_USER
from run import GROUPS

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
TOP_N = 15
SIGNAL = "uid"


def per_text_enrichment(model, tokenizer, texts, effective_ids, user_id_map, max_length, uid_len):
    device = next(model.parameters()).device
    out_rows = []

    model.eval()
    with torch.no_grad():
        for text, uid_key in zip(texts, effective_ids):
            enc = tokenizer(text, truncation=True, padding="max_length",
                            max_length=max_length, return_tensors="pt")
            input_ids = enc["input_ids"][0]
            attention_mask = enc["attention_mask"][0]

            uid_tokens = torch.tensor(user_id_map[uid_key], dtype=torch.long)
            input_ids = torch.cat([input_ids[:1], uid_tokens, input_ids[1:]])[:max_length]
            uid_mask = torch.ones(len(uid_tokens), dtype=torch.long)
            attention_mask = torch.cat([attention_mask[:1], uid_mask, attention_mask[1:]])[:max_length]

            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)

            out = model.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                output_attentions=True)

            valid = attention_mask[0].bool()
            seq_len = int(valid.sum().item())
            uid_start, uid_end = 1, min(1 + uid_len, seq_len)
            n_uid = max(uid_end - uid_start, 0)
            if n_uid == 0 or seq_len == 0:
                out_rows.append(None)
                continue

            layers = torch.stack([a[0] for a in out.attentions])       # [L, H, S, S]
            recv = (layers * valid.view(1, 1, -1, 1)).sum(dim=2)       # [L, H, S]
            recv = recv / recv.sum(dim=2, keepdim=True).clamp(min=1e-9)
            mass = recv[:, :, uid_start:uid_end].sum(dim=2)            # [L, H]

            out_rows.append((mass / (n_uid / seq_len)).cpu().numpy())

    return out_rows


def main():
    model, tokenizer, max_length, user_id_map, user_mapping = load_model(CHECKPOINT_PATH)
    uid_len = len(next(iter(user_id_map.values())))

    df = pd.read_csv(TEST_CSV).reset_index(drop=True)
    df["effective_id"] = [user_mapping.get(u, UNKNOWN_USER) for u in df["user_id"]]

    print(f"Verarbeite {len(df)} Texte ...")
    per_text = per_text_enrichment(model, tokenizer, df["text"].tolist(),
                                   df["effective_id"].tolist(), user_id_map,
                                   max_length, uid_len)

    valid_idx = [i for i, v in enumerate(per_text) if v is not None]
    stack = np.stack([per_text[i] for i in valid_idx])                 # [N, L, H]
    n_layers, n_heads = stack.shape[1], stack.shape[2]
    print(f"{n_layers} Layer x {n_heads} Heads, {len(valid_idx)} auswertbare Texte\n")

    rows = []
    for name, mask_fn in GROUPS:
        mask = mask_fn(df).to_numpy()
        sel = [k for k, i in enumerate(valid_idx) if mask[i]]
        if not sel:
            continue
        m = stack[sel].mean(axis=0)
        for l in range(n_layers):
            for h in range(n_heads):
                rows.append({"group": name, "layer": l, "head": h,
                             f"enrichment_{SIGNAL}": float(m[l, h])})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "head_enrichment_by_group.csv"), index=False)

    overall = stack.mean(axis=0)
    flat = pd.DataFrame(
        [{"layer": l, "head": h, f"enrichment_{SIGNAL}": float(overall[l, h])}
         for l in range(n_layers) for h in range(n_heads)]
    ).sort_values(f"enrichment_{SIGNAL}", ascending=False)
    flat.to_csv(os.path.join(OUT_DIR, "head_enrichment_overall.csv"), index=False)

    col = f"enrichment_{SIGNAL}"
    print(f"Mittel ueber alle Heads: {overall.mean():.3f}")
    print(f"Maximum:                 {overall.max():.3f}  "
          f"(Layer {int(flat.iloc[0]['layer'])}, Head {int(flat.iloc[0]['head'])})")
    for thr in (1.0, 2.0, 3.0, 5.0):
        print(f"  Heads > {thr}: {(overall > thr).sum():>3} von {overall.size}")

    print(f"\nTop {TOP_N} Heads (ueber alle Texte):")
    print(flat.head(TOP_N).to_string(index=False))

    print("\nMittelwert und Maximum pro Layer:")
    for l in range(n_layers):
        print(f"  Layer {l:>2}: mean {overall[l].mean():.3f}   max {overall[l].max():.3f} "
              f"(Head {int(overall[l].argmax())})")

    print("\nStaerkster Head pro Gruppe:")
    for name, _ in GROUPS:
        g = res[res["group"] == name]
        if g.empty:
            continue
        top = g.loc[g[col].idxmax()]
        print(f"  {name:<14} Layer {int(top['layer']):>2} Head {int(top['head']):>2}  "
              f"{top[col]:.3f}")

    print(f"\nErgebnisse in {OUT_DIR}/ gespeichert.")


if __name__ == "__main__":
    main()
