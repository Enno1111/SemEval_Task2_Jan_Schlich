"""
group_error_analysis.py
Baustein 1: gruppenweise Fehler-/Bias-Analyse. Nutzt ausschliesslich
predictions.csv und die Gruppen-Masken aus run.py, damit die Gruppen 1:1
mit den bereits berichteten r_composite-Tabellen uebereinstimmen.

Ausfuehren im Ordner 3_Regression_Combined: python group_error_analysis.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run import GROUPS, PREDICTIONS_CSV  # nur Modul-Level-Definitionen, kein run.main()!

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


def residual_stats(df, mask_fn, target):
    sub = df[mask_fn(df)]
    if len(sub) == 0:
        return {"n": 0, "bias": float("nan"), "mae": float("nan"), "rmse": float("nan"), "std": float("nan")}
    residual = sub[f"{target}_preds"] - sub[target]
    return {
        "n": len(sub),
        "bias": residual.mean(),        # systematische Ueber-/Unterschaetzung
        "mae": residual.abs().mean(),
        "rmse": float(np.sqrt((residual ** 2).mean())),
        "std": residual.std(),
    }


def main():
    df = pd.read_csv(PREDICTIONS_CSV)

    rows = []
    for name, mask_fn in GROUPS:
        for target in ("valence", "arousal"):
            stats = residual_stats(df, mask_fn, target)
            stats.update(group=name, target=target)
            rows.append(stats)

    summary = pd.DataFrame(rows)[["group", "target", "n", "bias", "mae", "rmse", "std"]]
    summary.to_csv(os.path.join(OUT_DIR, "group_error_summary.csv"), index=False)
    print(summary.to_string(index=False))

    for target in ("valence", "arousal"):
        abs_gold = df[target].abs()
        bins = pd.cut(abs_gold, bins=[-0.01, 0.5, 1.5, 2.5], labels=["low", "mid", "high"])
        by_extremity = df.groupby(bins, observed=True).apply(
            lambda g: (g[f"{target}_preds"] - g[target]).mean()
        )
        print(f"\nBias von {target} nach Extremitaet des Gold-Labels:\n{by_extremity}")

    for target in ("valence", "arousal"):
        fig, ax = plt.subplots(figsize=(8, 4))
        data, labels = [], []
        for name, mask_fn in GROUPS:
            sub = df[mask_fn(df)]
            if len(sub) == 0:
                continue
            data.append(sub[f"{target}_preds"] - sub[target])
            labels.append(name)
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_ylabel(f"Residual ({target}_pred - {target}_gold)")
        ax.set_title(f"Residualverteilung pro Gruppe -- {target}")
        plt.xticks(rotation=20)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"residuals_{target}.png"), dpi=150)
        plt.close(fig)

    print(f"\nPlots und Summary in {OUT_DIR}/ gespeichert.")


if __name__ == "__main__":
    main()
