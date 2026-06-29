import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from eval import task1_correlation

PREDICTIONS_CSV = "predictions.csv"   # liegt im Arbeitsverzeichnis, daher kein "../"

df = pd.read_csv(PREDICTIONS_CSV)

valence_scores = task1_correlation(
    user_ids=df["user_id"],
    text_ids=df["text_id"],
    predictions=df["valence_pred"],
    labels=df["valence"],
)

arousal_scores = task1_correlation(
    user_ids=df["user_id"],
    text_ids=df["text_id"],
    predictions=df["arousal_pred"],
    labels=df["arousal"],
)

print("Valence:", valence_scores)
print("Arousal:", arousal_scores)