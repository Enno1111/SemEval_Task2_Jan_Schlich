import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import model as train_module
import predict as predict_module
from eval import task1_correlation

SEEDS = [42, 123, 456]
PREDICTIONS_CSV = "predictions.csv"

def evaluate():
    df = pd.read_csv(PREDICTIONS_CSV)

    valence_scores = task1_correlation(
        user_ids=df["user_id"],
        text_ids=df["text_id"],
        predictions=df["valence_preds"],
        labels=df["valence"],
    )
    arousal_scores = task1_correlation(
        user_ids=df["user_id"],
        text_ids=df["text_id"],
        predictions=df["arousal_preds"],
        labels=df["arousal"],
    )

    return {
        "r_composite": (valence_scores["r_composite"] + arousal_scores["r_composite"]) / 2,
        "r_valence":    valence_scores["r_composite"],
        "r_arousal":    arousal_scores["r_composite"],
    }

def main():
    all_results = []

    for seed in SEEDS:
        print(f"\n{'='*40}")
        print(f"Seed {seed}")
        print(f"{'='*40}")

        # Seed in beiden Modulen als globale Variable setzen
        # -> train_test_split, set_seed(), Generator lesen ihn von dort
        train_module.SEED = seed
        predict_module.SEED = seed

        print("Training...")
        train_module.main()

        print("Predicting...")
        predict_module.main()

        print("Evaluating...")
        results = evaluate()
        all_results.append(results)

        print(f"r_composite: {results['r_composite']:.3f}")
        print(f"r_valence:   {results['r_valence']:.3f}")
        print(f"r_arousal:   {results['r_arousal']:.3f}")

    # Durchschnitt + Standardabweichung über alle Seeds
    print(f"\n{'='*40}")
    print(f"Ergebnis über {len(SEEDS)} Seeds")
    print(f"{'='*40}")
    for metric in ["r_composite", "r_valence", "r_arousal"]:
        values = [r[metric] for r in all_results]
        print(f"{metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")

if __name__ == "__main__":
    main()