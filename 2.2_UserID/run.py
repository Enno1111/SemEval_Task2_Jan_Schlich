"""
run.py
Trainiert/prediktet ueber SEEDS hinweg mit der aktuellen model.py-Config und
gibt r_composite/r_valence/r_arousal gemittelt mit Standardabweichung aus --
aufgeschluesselt nach:
  - seen vs. unseen User (Test-User, der auch im Training vorkam vs. nicht)
  - innerhalb der seen-Gruppe: User, die genug Trainings-Texte fuer eine
    eigene ID haben (>= MIN_USER_TEXTS), vs. User, die auf UNKNOWN
    zurueckfallen. Bei Modellen ohne User-IDs existiert diese Unterscheidung
    im Modell nicht -- dieselbe Aufteilung dient dort als Kontrollgruppe:
    zeigt der Baseline-Lauf denselben Abstand, stammt er aus der
    Datenzusammensetzung und nicht aus der Personalisierung.
  - is_words vs. not is_words
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
PREDICTIONS_CSV = "results/predictions.csv"

# Schwelle fuer "eigene ID": aus model.py, falls dort vorhanden (2.2/3),
# sonst der dort ueblich verwendete Wert -- so bleiben die Gruppen ueber
# alle Modelle hinweg vergleichbar.
MIN_USER_TEXTS = getattr(train_module, "MIN_USER_TEXTS", 15)

_train_counts = pd.read_csv(train_module.DATA_CSV).groupby("user_id").size()
OWN_ID_USERS = set(_train_counts[_train_counts >= MIN_USER_TEXTS].index)

GROUPS = [
    ("overall",      lambda df: df["user_id"].notna()),
    ("seen",         lambda df: df["is_seen_user"] == True),
    ("unseen",       lambda df: df["is_seen_user"] == False),
    ("seen_own_id",  lambda df: (df["is_seen_user"] == True) & df["user_id"].isin(OWN_ID_USERS)),
    ("seen_unknown", lambda df: (df["is_seen_user"] == True) & ~df["user_id"].isin(OWN_ID_USERS)),
    ("is_words",     lambda df: df["is_words"] == True),
    ("not_is_words", lambda df: df["is_words"] == False),
]

METRICS = ["r_composite", "r_valence", "r_arousal"]


def _scores(df):
    # Sehr kleine/degenerierte Gruppen koennen in eval.py zu nan oder einer
    # Division durch null fuehren -- dann nan zurueckgeben statt abzubrechen.
    try:
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
    except Exception:
        return {m: float("nan") for m in METRICS}

    return {
        "r_composite": (valence_scores["r_composite"] + arousal_scores["r_composite"]) / 2,
        "r_valence":    valence_scores["r_composite"],
        "r_arousal":    arousal_scores["r_composite"],
    }


def evaluate():
    df = pd.read_csv(PREDICTIONS_CSV)
    return {name: _scores(df[mask(df)]) for name, mask in GROUPS}


def print_group_composition():
    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"\n{'='*40}")
    print(f"Gruppen (MIN_USER_TEXTS = {MIN_USER_TEXTS})")
    print(f"{'='*40}")
    for name, mask in GROUPS:
        sub = df[mask(df)]
        print(f"  {name:<14} {len(sub):>5} texte, {sub['user_id'].nunique():>3} user")


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

        for name, _ in GROUPS:
            scores = results[name]
            print(f"  {name}: r_composite={scores['r_composite']:.3f} "
                  f"r_valence={scores['r_valence']:.3f} r_arousal={scores['r_arousal']:.3f}")

    print_group_composition()

    # Durchschnitt + Standardabweichung ueber alle Seeds, pro Gruppe
    print(f"\n{'='*40}")
    print(f"Ergebnis ueber {len(SEEDS)} Seeds")
    print(f"{'='*40}")

    for name, _ in GROUPS:
        print(f"\n{name}:")
        for metric in METRICS:
            values = [r[name][metric] for r in all_results]
            n_valid = int(np.sum(~np.isnan(values)))
            suffix = "" if n_valid == len(SEEDS) else f"   (nur {n_valid}/{len(SEEDS)} seeds gueltig)"
            print(f"  {metric}: {np.nanmean(values):.3f} ± {np.nanstd(values):.3f}{suffix}")

if __name__ == "__main__":
    main()
