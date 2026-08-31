"""
counterfactual_ablation.py
Baustein 2: Counterfactual Token-Ablation fuer das reine User-ID-Modell.
Ersetzt fuer jede Test-Zeile die echte User-ID durch den UNKNOWN-Platzhalter
und misst die Verschiebung der Vorhersage gegenueber der unveraenderten
Eingabe (= vorhandene predictions.csv).

Erwartung als Sanity-Check: seen_unknown und unseen sollten kaum eine
Verschiebung zeigen (die bekommen ohnehin schon immer UNKNOWN), nur
seen_own_id sollte einen echten Ausschlag zeigen.

Setzt einen bereits trainierten Checkpoint voraus (../models/dual_head_model_UserID.pt).
Ausfuehren im Ordner 2.2_UserID: python counterfactual_ablation.py
"""
import os
import pandas as pd
from torch.utils.data import DataLoader

from model import AffectDataset
from predict import load_model, predict, CHECKPOINT_PATH, TEST_CSV, BATCH_SIZE, UNKNOWN_USER
from run import GROUPS

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
ORIGINAL_PREDICTIONS_CSV = "results/predictions.csv"


def build_loader(texts, effective_ids, user_id_map, tokenizer, max_length):
    dummy = [0] * len(texts)
    ds = AffectDataset(texts, dummy, dummy, tokenizer, max_length, effective_ids, user_id_map)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


def main():
    model, tokenizer, max_length, user_id_map, user_mapping = load_model(CHECKPOINT_PATH)

    df = pd.read_csv(TEST_CSV)
    texts = df["text"].tolist()
    user_ids = df["user_id"].tolist()
    effective_ids_unknown = [UNKNOWN_USER] * len(user_ids)

    orig = pd.read_csv(ORIGINAL_PREDICTIONS_CSV)[["text_id", "valence_preds", "arousal_preds"]]
    orig = orig.rename(columns={"valence_preds": "valence_preds_orig", "arousal_preds": "arousal_preds_orig"})

    print("Ablation: User-ID -> UNKNOWN ...")
    v_user, a_user = predict(model, build_loader(texts, effective_ids_unknown, user_id_map, tokenizer, max_length))

    df["valence_preds_user_ablated"] = v_user
    df["arousal_preds_user_ablated"] = a_user
    df = df.merge(orig, on="text_id", how="left")

    df["delta_user_valence"] = (df["valence_preds_user_ablated"] - df["valence_preds_orig"]).abs()
    df["delta_user_arousal"] = (df["arousal_preds_user_ablated"] - df["arousal_preds_orig"]).abs()
    df.to_csv(os.path.join(OUT_DIR, "counterfactual_ablation_full.csv"), index=False)

    rows = []
    for name, mask_fn in GROUPS:
        sub = df[mask_fn(df)]
        if len(sub) == 0:
            continue
        rows.append({
            "group": name, "n": len(sub),
            "mean_|delta_user|_valence": sub["delta_user_valence"].mean(),
            "mean_|delta_user|_arousal": sub["delta_user_arousal"].mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "counterfactual_ablation_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
