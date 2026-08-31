"""
counterfactual_ablation.py
Baustein 2: Counterfactual Token-Ablation. Testet, wie stark sich die
Vorhersage veraendert, wenn man
  (a) den User-Identifier-Block durch den UNKNOWN-Platzhalter ersetzt
  (b) den Temporal-Prefix entfernt
verglichen mit der unveraenderten Eingabe (= vorhandene predictions.csv).

Erwartung als Sanity-Check: seen_unknown und unseen sollten bei delta_user
nahe null liegen (die bekommen ohnehin schon immer UNKNOWN), nur
seen_own_id sollte einen echten Ausschlag zeigen.

Setzt einen bereits trainierten Checkpoint voraus (../models/dual_head_model_all.pt).
Ausfuehren im Ordner 3_Regression_Combined: python counterfactual_ablation.py
"""
import os
import pandas as pd
from torch.utils.data import DataLoader

from model import AffectDataset, format as date_format
from predict import load_model, predict, CHECKPOINT_PATH, TEST_CSV, BATCH_SIZE, UNKNOWN_USER
from run import GROUPS

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
ORIGINAL_PREDICTIONS_CSV = "results/predictions.csv"

# Statt den Prefix ersatzlos zu loeschen, wird ein struktur-identischer
# Platzhalter eingesetzt: gleiche Form, gleiche Tokenanzahl, aber ohne
# Datumsinformation. Das haelt die Sequenzlaenge konstant und vermeidet einen
# Input, den das Modell im Training nie gesehen hat -- analog zu UNKNOWN_USER.
PLACEHOLDER_PREFIX = "year: 0000 month: 00 day: 00"


def build_loader(texts, effective_ids, user_id_map, tokenizer, max_length):
    dummy = [0] * len(texts)
    ds = AffectDataset(texts, dummy, dummy, tokenizer, max_length, effective_ids, user_id_map)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


def main():
    model, tokenizer, max_length, user_id_map, user_mapping = load_model(CHECKPOINT_PATH)

    df = pd.read_csv(TEST_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["time_str"] = df["timestamp"].dt.strftime(date_format)

    texts_full    = (df["time_str"] + " " + df["text"]).tolist()
    texts_no_date = (PLACEHOLDER_PREFIX + " " + df["text"]).tolist()

    user_ids = df["user_id"].tolist()
    effective_ids_real    = [user_mapping.get(uid, UNKNOWN_USER) for uid in user_ids]
    effective_ids_unknown = [UNKNOWN_USER] * len(user_ids)

    orig = pd.read_csv(ORIGINAL_PREDICTIONS_CSV)[["text_id", "valence_preds", "arousal_preds"]]
    orig = orig.rename(columns={"valence_preds": "valence_preds_orig", "arousal_preds": "arousal_preds_orig"})

    print("Ablation (a): User-ID -> UNKNOWN ...")
    v_user, a_user = predict(model, build_loader(texts_full, effective_ids_unknown, user_id_map, tokenizer, max_length))

    print(f"Ablation (b): Datums-Prefix -> {PLACEHOLDER_PREFIX!r} ...")
    v_date, a_date = predict(model, build_loader(texts_no_date, effective_ids_real, user_id_map, tokenizer, max_length))

    df["valence_preds_user_ablated"] = v_user
    df["arousal_preds_user_ablated"] = a_user
    df["valence_preds_date_ablated"] = v_date
    df["arousal_preds_date_ablated"] = a_date
    df = df.merge(orig, on="text_id", how="left")

    df["delta_user_valence"] = (df["valence_preds_user_ablated"] - df["valence_preds_orig"]).abs()
    df["delta_user_arousal"] = (df["arousal_preds_user_ablated"] - df["arousal_preds_orig"]).abs()
    df["delta_date_valence"] = (df["valence_preds_date_ablated"] - df["valence_preds_orig"]).abs()
    df["delta_date_arousal"] = (df["arousal_preds_date_ablated"] - df["arousal_preds_orig"]).abs()
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
            "mean_|delta_date|_valence": sub["delta_date_valence"].mean(),
            "mean_|delta_date|_arousal": sub["delta_date_arousal"].mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "counterfactual_ablation_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
