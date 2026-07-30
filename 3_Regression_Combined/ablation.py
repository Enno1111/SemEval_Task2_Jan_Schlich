"""
fully vibe coded
ablation.py
Führt eine sequentielle Ablation-Studie durch. Der erste Schritt testet
verschiedene Encoder-Modelle (mit ggf. abweichender Learning Rate pro
Modell), danach folgen die übrigen Hyperparameter auf dem Sieger-Modell.
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

# Kandidaten für MODEL_NAME mit optionalen Parameter-Overrides
# (z.B. abweichende Learning Rate für bereits emotion-finegetunte Modelle)
MODEL_CANDIDATES = [
    {"MODEL_NAME": "bert-base-uncased"},
    {"MODEL_NAME": "roberta-base"},
    {"MODEL_NAME": "cardiffnlp/twitter-roberta-base-sentiment-latest"},
    {"MODEL_NAME": "cardiffnlp/twitter-roberta-base-emotion"},
    {"MODEL_NAME": "answerdotai/ModernBERT-base"},
    {"MODEL_NAME": "microsoft/deberta-base-mnli"},
    {"MODEL_NAME": "distilroberta-base",              "LEARNING_RATE": 1e-5},
    {"MODEL_NAME": "SamLowe/roberta-base-go_emotions", "LEARNING_RATE": 1e-5},
    {"MODEL_NAME": "RobroKools/vad-bert",              "LEARNING_RATE": 1e-5},
]

# Restliche Ablation-Schritte, in dieser Reihenfolge nach dem Modell-Schritt
ABLATION_STEPS = [
    ("POOLING_STRATEGY", ["cls", "mean"]),
    ("HEAD_HIDDEN_SIZE", [None, 128, 256]),
    ("LEARNING_RATE",    [2e-5, 3e-5, 5e-5]),
    ("BATCH_SIZE",       [16, 32]),
    ("DROPOUT",          [0.1, 0.2]),
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
    # Startkonfiguration = aktuelle Werte in model.py
    best_config = {
        "MODEL_NAME":       train_module.MODEL_NAME,
        "POOLING_STRATEGY": train_module.POOLING_STRATEGY,
        "HEAD_HIDDEN_SIZE": train_module.HEAD_HIDDEN_SIZE,
        "LEARNING_RATE":    train_module.LEARNING_RATE,
        "BATCH_SIZE":       train_module.BATCH_SIZE,
        "DROPOUT":          train_module.DROPOUT,
    }

    all_step_results = {}
    winners = {}

    # ---------------------------------------------------------
    # Schritt 0: MODEL_NAME (mit ggf. abweichender LR pro Modell)
    # ---------------------------------------------------------
    print(f"\n{'#'*50}")
    print(f"# Ablation: MODEL_NAME")
    print(f"{'#'*50}")

    model_step_results = []

    for candidate in MODEL_CANDIDATES:
        # zurück zur Basiskonfiguration, dann Overrides anwenden
        for k, v in best_config.items():
            setattr(train_module, k, v)

        for k, v in candidate.items():
            setattr(train_module, k, v)

        model_name = candidate["MODEL_NAME"]
        used_lr = candidate.get("LEARNING_RATE", best_config["LEARNING_RATE"])

        print(f"\n--- MODEL_NAME = {model_name} (LR={used_lr}) ---")
        result = run_seeds()
        result["value"] = candidate   # gesamtes Dict speichern, nicht nur den Namen
        model_step_results.append(result)

        print_result(f"{model_name}", result)

    best_model_result = max(model_step_results, key=lambda r: r["r_composite"])
    best_config.update(best_model_result["value"])   # MODEL_NAME + ggf. LEARNING_RATE übernehmen
    all_step_results["MODEL_NAME"] = model_step_results
    winners["MODEL_NAME"] = best_model_result

    print(f"\n>>> Bestes Modell: {best_model_result['value']['MODEL_NAME']}")
    print_result("Bestes Modell", best_model_result)

    # ---------------------------------------------------------
    # Schritte 1-5: restliche Hyperparameter auf dem Sieger-Modell
    # ---------------------------------------------------------
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
        label = result["value"]["MODEL_NAME"] if param_name == "MODEL_NAME" else result["value"]
        print(f"\n{param_name} = {label}")
        print(f"  r_composite: {result['r_composite']:.3f} ± {result['r_composite_std']:.3f}")
        print(f"  r_valence:   {result['r_valence']:.3f} ± {result['r_valence_std']:.3f}")
        print(f"  r_arousal:   {result['r_arousal']:.3f} ± {result['r_arousal_std']:.3f}")


if __name__ == "__main__":
    main()