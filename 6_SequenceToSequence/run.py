"""
fully vibe coded
run.py
Trainiert/prediktet ueber SEEDS hinweg mit der aktuellen model.py-Config und
gibt r_composite/r_valence/r_arousal gemittelt mit Standardabweichung aus --
aufgeschluesselt nach seen vs. unseen User und is_words vs. not is_words.
"""

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

GROUPS = [
    ("seen",         lambda df: df["is_seen_user"] == True),
    ("unseen",       lambda df: df["is_seen_user"] == False),
    ("is_words",     lambda df: df["is_words"] == True),
    ("not_is_words", lambda df: df["is_words"] == False),
]


def _scores(df):
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


def evaluate():
    df = pd.read_csv(PREDICTIONS_CSV)
    return {name: _scores(df[mask(df)]) for name, mask in GROUPS}


def print_group_scores(results):
    for name, _ in GROUPS:
        scores = results[name]
        print(f"  {name}:")
        print(f"    r_composite: {scores['r_composite']:.3f}")
        print(f"    r_valence:   {scores['r_valence']:.3f}")
        print(f"    r_arousal:   {scores['r_arousal']:.3f}")


def main():
    all_results = []

    for seed in SEEDS:
        print(f"\n{'='*40}")
        print(f"Seed {seed}")
        print(f"{'='*40}")

        # Seed in beiden Modulen als globale Variable setzen
        # -> train_test_split, set_seed(), generate_user_identifiers() lesen ihn von dort
        train_module.SEED = seed
        predict_module.SEED = seed

        print("Training...")
        train_module.main()

        print("Predicting...")
        predict_module.main()

        print("Evaluating...")
        results = evaluate()
        all_results.append(results)

        print_group_scores(results)

    # Durchschnitt + Standardabweichung ueber alle Seeds, pro Gruppe
    print(f"\n{'='*40}")
    print(f"Ergebnis ueber {len(SEEDS)} Seeds")
    print(f"{'='*40}")

    for name, _ in GROUPS:
        print(f"\n{name}:")
        for metric in ["r_composite", "r_valence", "r_arousal"]:
            values = [r[name][metric] for r in all_results]
            print(f"  {metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")

if __name__ == "__main__":
    main()
