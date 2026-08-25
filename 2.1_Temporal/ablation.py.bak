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

# Temporal prepending strategies to evaluate
TEMPORAL_MODES = ["none", "wave", "date", "hour", "full"]


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


def run_seeds():
    all_results = []

    for seed in SEEDS:
        train_module.SEED = seed
        predict_module.SEED = seed

        train_module.main()
        predict_module.main()

        results = evaluate()
        all_results.append(results)

    r_composite_vals = [r["r_composite"] for r in all_results]
    r_valence_vals   = [r["r_valence"] for r in all_results]
    r_arousal_vals   = [r["r_arousal"] for r in all_results]

    return {
        "r_composite":     np.mean(r_composite_vals),
        "r_composite_std": np.std(r_composite_vals),
        "r_valence":       np.mean(r_valence_vals),
        "r_valence_std":   np.std(r_valence_vals),
        "r_arousal":       np.mean(r_arousal_vals),
        "r_arousal_std":   np.std(r_arousal_vals),
    }


def print_result(label, result):
    print(f"{label}")
    print(f"  r_composite: {result['r_composite']:.3f} ± {result['r_composite_std']:.3f}")
    print(f"  r_valence:   {result['r_valence']:.3f} ± {result['r_valence_std']:.3f}")
    print(f"  r_arousal:   {result['r_arousal']:.3f} ± {result['r_arousal_std']:.3f}")


def main():
    print(f"\n{'#'*50}")
    print(f"# Ablation: TEMPORAL_MODE")
    print(f"{'#'*50}")

    step_results = []

    for value in TEMPORAL_MODES:
        train_module.TEMPORAL_MODE = value
        predict_module.TEMPORAL_MODE = value

        print(f"\n--- TEMPORAL_MODE = {value} ---")
        result = run_seeds()
        result["value"] = value
        step_results.append(result)

        print_result(f"TEMPORAL_MODE={value}", result)

    best_result = max(step_results, key=lambda r: r["r_composite"])

    print(f"\n>>> Bester Wert für TEMPORAL_MODE: {best_result['value']}")
    print_result("Sieger TEMPORAL_MODE", best_result)

    print(f"\n{'='*50}")
    print("Finale beste Konfiguration")
    print(f"{'='*50}")
    print(f"TEMPORAL_MODE: {best_result['value']}")

    print(f"\n{'='*50}")
    print("Alle getesteten Werte — Übersicht")
    print(f"{'='*50}")
    for result in step_results:
        print_result(f"TEMPORAL_MODE={result['value']}", result)


if __name__ == "__main__":
    main()