"""
fully vibe coded
ablation.py
Testet CHUNK_SIZE und MIN_USER_TEXTS fuer die User-Sequenz-Architektur
(mehrere Texte desselben Users in einer Sequenz, getrennt durch [CLS]/[SEP],
UNK-Pool fuer User unterhalb der Schwelle). Zusaetzlich zu den ueblichen
Metriken wird bei jedem Schritt r_composite getrennt fuer Test-User, die
bereits im Training vorkamen ("seen"), und fuer komplett neue Test-User
("unseen") ausgegeben.
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

ABLATION_STEPS = [
    ("CHUNK_SIZE",      [3, 5, 8, 10]),
    ("MIN_USER_TEXTS",  [3, 5, 10, 15]),
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

    overall = _scores(df)
    seen    = _scores(df[df["seen"] == True])
    unseen  = _scores(df[df["seen"] == False])

    return {"overall": overall, "seen": seen, "unseen": unseen}


def run_seeds():
    all_results = []

    for seed in SEEDS:
        train_module.SEED = seed
        predict_module.SEED = seed

        train_module.main()
        predict_module.main()

        results = evaluate()
        all_results.append(results)

    def aggregate(group):
        r_composite_vals = [r[group]["r_composite"] for r in all_results]
        r_valence_vals   = [r[group]["r_valence"] for r in all_results]
        r_arousal_vals   = [r[group]["r_arousal"] for r in all_results]
        return {
            "r_composite":     np.mean(r_composite_vals),
            "r_composite_std": np.std(r_composite_vals),
            "r_valence":       np.mean(r_valence_vals),
            "r_valence_std":   np.std(r_valence_vals),
            "r_arousal":       np.mean(r_arousal_vals),
            "r_arousal_std":   np.std(r_arousal_vals),
        }

    return {
        "overall": aggregate("overall"),
        "seen":    aggregate("seen"),
        "unseen":  aggregate("unseen"),
    }


def print_block(label, scores):
    print(f"  {label}:")
    print(f"    r_composite: {scores['r_composite']:.3f} ± {scores['r_composite_std']:.3f}")
    print(f"    r_valence:   {scores['r_valence']:.3f} ± {scores['r_valence_std']:.3f}")
    print(f"    r_arousal:   {scores['r_arousal']:.3f} ± {scores['r_arousal_std']:.3f}")


def print_result(label, result):
    print(f"{label}")
    print_block("overall", result["overall"])
    print_block("seen (User bereits in Train)", result["seen"])
    print_block("unseen (User komplett neu)", result["unseen"])


def main():
    # Startkonfiguration = aktuelle Werte in model.py
    best_config = {
        "CHUNK_SIZE":     train_module.CHUNK_SIZE,
        "MIN_USER_TEXTS": train_module.MIN_USER_TEXTS,
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

            print(f"\n--- {param_name} = {value} ---")
            result = run_seeds()
            result["value"] = value
            step_results.append(result)

            print_result(f"{param_name}={value}", result)

        # Auswahl des Siegers weiterhin anhand von overall/r_composite
        best_result = max(step_results, key=lambda r: r["overall"]["r_composite"])
        best_config[param_name] = best_result["value"]
        all_step_results[param_name] = step_results
        winners[param_name] = best_result

        print(f"\n>>> Bester Wert fuer {param_name}: {best_result['value']}")
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
    print("Sieger pro Parameter — alle Metriken (overall/seen/unseen)")
    print(f"{'='*50}")
    for param_name, result in winners.items():
        print(f"\n{param_name} = {result['value']}")
        print_result("", result)


if __name__ == "__main__":
    main()
