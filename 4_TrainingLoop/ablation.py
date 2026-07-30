"""
fully vibe coded
ablation.py
Testet verschiedene Trainings-Loop-Konfigurationen (Scheduler, Warmup,
Gradient-Clipping, Loss-Gewichtung) auf Basis der aktuellen model.py-Werte.
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

# Kandidaten für den Trainings-Loop: Scheduler-Typ, Warmup-Anteil,
# Gradient-Clipping-Norm und Loss-Gewichtung pro Kopf
TRAINING_LOOP_CANDIDATES = [
    {"name": "baseline (linear, no warmup)",
     "SCHEDULER_TYPE": "linear", "WARMUP_RATIO": 0.0, "GRAD_CLIP_NORM": None,
     "VALENCE_LOSS_WEIGHT": 1.0, "AROUSAL_LOSS_WEIGHT": 1.0},

    {"name": "linear + 10% warmup",
     "SCHEDULER_TYPE": "linear", "WARMUP_RATIO": 0.1, "GRAD_CLIP_NORM": None,
     "VALENCE_LOSS_WEIGHT": 1.0, "AROUSAL_LOSS_WEIGHT": 1.0},

    {"name": "cosine + 10% warmup",
     "SCHEDULER_TYPE": "cosine", "WARMUP_RATIO": 0.1, "GRAD_CLIP_NORM": None,
     "VALENCE_LOSS_WEIGHT": 1.0, "AROUSAL_LOSS_WEIGHT": 1.0},

    {"name": "constant + 10% warmup",
     "SCHEDULER_TYPE": "constant", "WARMUP_RATIO": 0.1, "GRAD_CLIP_NORM": None,
     "VALENCE_LOSS_WEIGHT": 1.0, "AROUSAL_LOSS_WEIGHT": 1.0},

    {"name": "linear + warmup + grad clip 1.0",
     "SCHEDULER_TYPE": "linear", "WARMUP_RATIO": 0.1, "GRAD_CLIP_NORM": 1.0,
     "VALENCE_LOSS_WEIGHT": 1.0, "AROUSAL_LOSS_WEIGHT": 1.0},

    {"name": "linear + warmup, valence-lastiger Loss",
     "SCHEDULER_TYPE": "linear", "WARMUP_RATIO": 0.1, "GRAD_CLIP_NORM": None,
     "VALENCE_LOSS_WEIGHT": 1.5, "AROUSAL_LOSS_WEIGHT": 1.0},
]


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
    print(f"# Ablation: TRAINING_LOOP")
    print(f"{'#'*50}")

    step_results = []

    for candidate in TRAINING_LOOP_CANDIDATES:
        for k, v in candidate.items():
            if k == "name":
                continue
            setattr(train_module, k, v)

        print(f"\n--- TRAINING_LOOP = {candidate['name']} ---")
        result = run_seeds()
        result["value"] = candidate
        step_results.append(result)

        print_result(candidate["name"], result)

    best_result = max(step_results, key=lambda r: r["r_composite"])

    print(f"\n>>> Beste Trainings-Loop-Konfiguration: {best_result['value']['name']}")
    print_result(f"Sieger TRAINING_LOOP", best_result)

    print(f"\n{'='*50}")
    print("Finale beste Konfiguration")
    print(f"{'='*50}")
    for k, v in best_result["value"].items():
        print(f"{k}: {v}")

    print(f"\n{'='*50}")
    print("Alle getesteten Konfigurationen — Übersicht")
    print(f"{'='*50}")
    for result in step_results:
        print_result(result["value"]["name"], result)


if __name__ == "__main__":
    main()