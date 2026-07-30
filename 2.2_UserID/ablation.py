"""
fully vibe coded
ablation.py
Führt eine sequentielle Ablation-Studie für die UserID-Parameter durch:
zuerst MIN_USER_TEXTS (aktuell bester Wert: 15), danach USER_ID_LENGTH
(aktuell bester Wert: 3) auf Basis des Siegerwerts aus Schritt 1.
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

# Ablation-Schritte, in dieser Reihenfolge
ABLATION_STEPS = [
    ("MIN_USER_TEXTS",  [5, 10, 15, 20, 30]),
    ("USER_ID_LENGTH",  [1, 2, 3, 5, 8]),
]


def set_user_id_length(value):
    # generate_user_identifiers() bindet L=USER_ID_LENGTH als Default-Parameter
    # beim Import von model.py. train_module.USER_ID_LENGTH allein zu setzen hat
    # daher keinen Effekt auf main() — der Funktions-Default muss mitgepatcht werden.
    defaults = list(train_module.generate_user_identifiers.__defaults__)
    defaults[0] = value
    train_module.generate_user_identifiers.__defaults__ = tuple(defaults)


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
    # Startkonfiguration = aktuelle Werte in model.py
    best_config = {
        "MIN_USER_TEXTS": train_module.MIN_USER_TEXTS,
        "USER_ID_LENGTH": train_module.USER_ID_LENGTH,
    }

    all_step_results = {}
    winners = {}

    for param_name, candidate_values in ABLATION_STEPS:
        print(f"\n{'#'*50}")
        print(f"# Ablation: {param_name}")
        print(f"{'#'*50}")

        step_results = []

        for value in candidate_values:
            for k, v in best_config.items():
                setattr(train_module, k, v)

            setattr(train_module, param_name, value)
            if param_name == "USER_ID_LENGTH":
                set_user_id_length(value)

            print(f"\n--- {param_name} = {value} ---")
            result = run_seeds()
            result["value"] = value
            step_results.append(result)

            print_result(f"{param_name}={value}", result)

        best_result = max(step_results, key=lambda r: r["r_composite"])
        best_config[param_name] = best_result["value"]
        all_step_results[param_name] = step_results
        winners[param_name] = best_result

        print(f"\n>>> Bester Wert für {param_name}: {best_result['value']}")
        print_result(f"Sieger {param_name}", best_result)

    # ---------------------------------------------------------
    # Zusammenfassung
    # ---------------------------------------------------------
    print(f"\n{'='*50}")
    print("Finale beste Konfiguration")
    print(f"{'='*50}")
    for param_name, value in best_config.items():
        print(f"{param_name}: {value}")

    print(f"\n{'='*50}")
    print("Sieger pro Parameter — alle Metriken")
    print(f"{'='*50}")
    for param_name, result in winners.items():
        print(f"\n{param_name} = {result['value']}")
        print(f"  r_composite: {result['r_composite']:.3f} ± {result['r_composite_std']:.3f}")
        print(f"  r_valence:   {result['r_valence']:.3f} ± {result['r_valence_std']:.3f}")
        print(f"  r_arousal:   {result['r_arousal']:.3f} ± {result['r_arousal_std']:.3f}")


if __name__ == "__main__":
    main()
