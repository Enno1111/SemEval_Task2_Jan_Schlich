"""
counterfactual_ablation.py
Baustein 2: Counterfactual Token-Ablation fuer das reine Temporal-Modell.
Entfernt fuer jede Test-Zeile den Datums-Prefix ("year: ... month: ...
day: ...") und misst die Verschiebung der Vorhersage gegenueber der
unveraenderten Eingabe (= vorhandene predictions.csv).

Setzt einen bereits trainierten Checkpoint voraus (../models/dual_head_model_Temporal.pt).
Ausfuehren im Ordner 2.1_Temporal: python counterfactual_ablation.py
"""
import os
import pandas as pd
from torch.utils.data import DataLoader

from model import AffectDataset
from predict import load_model, predict, load_test_data, CHECKPOINT_PATH, TEST_CSV, BATCH_SIZE
from run import GROUPS

OUT_DIR = "explainability_out"
os.makedirs(OUT_DIR, exist_ok=True)
ORIGINAL_PREDICTIONS_CSV = "predictions.csv"

# Statt den Prefix ersatzlos zu loeschen, wird ein struktur-identischer
# Platzhalter eingesetzt: gleiche Form, gleiche Tokenanzahl, aber ohne
# Datumsinformation. Das haelt die Sequenzlaenge konstant und vermeidet einen
# Input, den das Modell im Training nie gesehen hat -- analog zu UNKNOWN_USER.
PLACEHOLDER_PREFIX = "year: 0000 month: 00 day: 00"


def build_loader(texts, tokenizer, max_length):
    dummy = [0] * len(texts)
    ds = AffectDataset(texts, dummy, dummy, tokenizer, max_length)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


def main():
    model, tokenizer, max_length = load_model(CHECKPOINT_PATH)

    # load_test_data liefert Texte MIT Datums-Prefix (wie predict.py sie nutzt) + df mit rohem 'text'
    _, _, _, df = load_test_data(TEST_CSV)
    texts_no_date = (PLACEHOLDER_PREFIX + " " + df["text"]).tolist()

    orig = pd.read_csv(ORIGINAL_PREDICTIONS_CSV)[["text_id", "valence_preds", "arousal_preds"]]
    orig = orig.rename(columns={"valence_preds": "valence_preds_orig", "arousal_preds": "arousal_preds_orig"})

    print(f"Ablation: Datums-Prefix -> '{PLACEHOLDER_PREFIX}' ...")
    v_date, a_date = predict(model, build_loader(texts_no_date, tokenizer, max_length))

    df["valence_preds_date_ablated"] = v_date
    df["arousal_preds_date_ablated"] = a_date
    df = df.merge(orig, on="text_id", how="left")

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
            "mean_|delta_date|_valence": sub["delta_date_valence"].mean(),
            "mean_|delta_date|_arousal": sub["delta_date_arousal"].mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "counterfactual_ablation_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
