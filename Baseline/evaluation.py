import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from eval import task1_correlation

PREDICTIONS_CSV = "predictions.csv"

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

for i,j in valence_scores.items():
    print(f"{i}: {round(j, 2)}")

for i,j in arousal_scores.items():
    print(f"{i}: {round(j, 2)}")

print(f"r_composite:{(round(valence_scores['r_composite'], 2) + round(arousal_scores['r_composite'], 2)) / 2}")