"""
head_diagnostic.py
Schnelldiagnose: Gibt es einzelne Attention-Heads, die sich auf die
User-ID-Tokens spezialisiert haben?

Berechnet die laengennormalisierte Enrichment auf den UID-Positionen
getrennt fuer jeden (Layer, Head) statt nur fuer den letzten Layer
gemittelt. Ein Wert von 1.0 entspricht dem Anteil, den die Tokens allein
aufgrund ihrer Anzahl erhalten wuerden.

Bewusst als Stichprobe angelegt (N_SAMPLE Texte), um schnell zu sein --
es geht nur um die Frage, ob sich eine genauere Untersuchung lohnt.

Ausfuehren im Ordner 2.2_UserID: python head_diagnostic.py
"""
import os

import pandas as pd
import torch

from predict import load_model, CHECKPOINT_PATH, TEST_CSV, UNKNOWN_USER

OUT_DIR = "explainability_out"
os.makedirs(OUT_DIR, exist_ok=True)
N_SAMPLE = 200
TOP_N = 15


def main():
    model, tokenizer, max_length, user_id_map, user_mapping = load_model(CHECKPOINT_PATH)
    uid_len = len(next(iter(user_id_map.values())))
    device = next(model.parameters()).device

    df = pd.read_csv(TEST_CSV)
    df["effective_id"] = [user_mapping.get(u, UNKNOWN_USER) for u in df["user_id"]]
    # nur Texte mit echter, individueller ID -- dort ist der Effekt zu erwarten
    df = df[df["effective_id"] != UNKNOWN_USER]
    if len(df) > N_SAMPLE:
        df = df.sample(n=N_SAMPLE, random_state=42)
    print(f"Stichprobe: {len(df)} Texte mit individueller User-ID")

    totals = None
    n = 0

    model.eval()
    with torch.no_grad():
        for text, uid_key in zip(df["text"].tolist(), df["effective_id"].tolist()):
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
                continue
            expected = n_uid / seq_len

            # [layers, heads]
            layers = torch.stack([a[0] for a in out.attentions])          # [L, H, S, S]
            recv = (layers * valid.view(1, 1, -1, 1)).sum(dim=2)          # [L, H, S]
            recv = recv / recv.sum(dim=2, keepdim=True).clamp(min=1e-9)
            mass = recv[:, :, uid_start:uid_end].sum(dim=2)               # [L, H]

            enrich = (mass / expected).cpu()
            totals = enrich if totals is None else totals + enrich
            n += 1

    mean = (totals / n).numpy()
    n_layers, n_heads = mean.shape

    rows = [{"layer": l, "head": h, "enrichment_uid": float(mean[l, h])}
            for l in range(n_layers) for h in range(n_heads)]
    res = pd.DataFrame(rows).sort_values("enrichment_uid", ascending=False)
    res.to_csv(os.path.join(OUT_DIR, "head_diagnostic.csv"), index=False)

    print(f"\n{n_layers} Layer x {n_heads} Heads, gemittelt ueber {n} Texte")
    print(f"Mittel ueber alle Heads: {mean.mean():.3f}")
    print(f"Maximum:                 {mean.max():.3f}  "
          f"(Layer {int(res.iloc[0]['layer'])}, Head {int(res.iloc[0]['head'])})")
    for thr in (1.0, 2.0, 5.0):
        print(f"Heads mit Enrichment > {thr}: {(mean > thr).sum()} von {mean.size}")

    print(f"\nTop {TOP_N} Heads:")
    print(res.head(TOP_N).to_string(index=False))

    print("\nMittelwert pro Layer:")
    for l in range(n_layers):
        print(f"  Layer {l:>2}: {mean[l].mean():.3f}   max {mean[l].max():.3f}")

    print(f"\nGespeichert in {OUT_DIR}/head_diagnostic.csv")


if __name__ == "__main__":
    main()
