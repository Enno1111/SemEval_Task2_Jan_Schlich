"""
offset_analysis.py
Testet, ob der User-Identifier nur einen konstanten Offset pro Nutzer lernt
oder die Vorhersage kontextabhaengig moduliert. Grundlage ist das signierte
delta_user pro Text aus counterfactual_ablation_full.csv.

Ein reiner Offset impliziert delta_u,t = c_u fuer alle Texte t eines Nutzers,
d.h. die Within-User-Varianz von delta waere null. Die Varianzzerlegung
quantifiziert, welcher Anteil der Gesamtvarianz auf konstante Nutzer-Offsets
(between) und welcher auf kontextabhaengige Modulation (within) entfaellt.

Ausfuehren im jeweiligen Experiment-Ordner: python offset_analysis.py
"""
import os

import numpy as np
import pandas as pd

IN_CSV = os.path.join("explainability_out", "counterfactual_ablation_full.csv")
OUT_CSV = os.path.join("explainability_out", "offset_analysis_summary.csv")
MIN_TEXTS_PER_USER = 3


def decompose(df, delta_col):
    grand_mean = df[delta_col].mean()

    per_user = df.groupby("user_id")[delta_col].agg(["mean", "std", "count"])
    per_user = per_user[per_user["count"] >= MIN_TEXTS_PER_USER]

    between_ss = (per_user["count"] * (per_user["mean"] - grand_mean) ** 2).sum()
    within_ss = df.groupby("user_id")[delta_col].transform(
        lambda g: (g - g.mean()) ** 2
    )
    within_ss = within_ss[df["user_id"].isin(per_user.index)].sum()
    total_ss = between_ss + within_ss

    return {
        "n_users": len(per_user),
        "n_texts": int(per_user["count"].sum()),
        "mean_abs_delta": df.loc[df["user_id"].isin(per_user.index), delta_col].abs().mean(),
        "mean_abs_user_offset": per_user["mean"].abs().mean(),
        "mean_within_user_sd": per_user["std"].mean(),
        "between_user_share": between_ss / total_ss if total_ss > 0 else float("nan"),
        "within_user_share": within_ss / total_ss if total_ss > 0 else float("nan"),
    }


def main():
    df = pd.read_csv(IN_CSV)

    delta_cols = [c for c in df.columns if c.startswith("delta_user_")]
    if not delta_cols:
        raise SystemExit(f"Keine delta_user_* Spalten in {IN_CSV} gefunden.")

    active = df[df[delta_cols].abs().max(axis=1) > 1e-10].copy()
    print(f"Texte mit aktivem User-Identifier: {len(active)} von {len(df)}")
    print(f"Nutzer mit aktivem Identifier: {active['user_id'].nunique()}")
    print(f"Mindestens {MIN_TEXTS_PER_USER} Texte pro Nutzer gefordert.\n")

    rows = []
    for col in delta_cols:
        target = col.replace("delta_user_", "")
        stats = decompose(active, col)
        stats["target"] = target
        rows.append(stats)

        print(f"--- {target} ---")
        print(f"  Nutzer / Texte:            {stats['n_users']} / {stats['n_texts']}")
        print(f"  mittleres |delta|:         {stats['mean_abs_delta']:.4f}")
        print(f"  mittlerer |Nutzer-Offset|: {stats['mean_abs_user_offset']:.4f}")
        print(f"  mittlere Within-User-SD:   {stats['mean_within_user_sd']:.4f}")
        print(f"  Varianzanteil between:     {stats['between_user_share']*100:.1f}%")
        print(f"  Varianzanteil within:      {stats['within_user_share']*100:.1f}%\n")

    summary = pd.DataFrame(rows)[
        [
            "target",
            "n_users",
            "n_texts",
            "mean_abs_delta",
            "mean_abs_user_offset",
            "mean_within_user_sd",
            "between_user_share",
            "within_user_share",
        ]
    ]
    summary.to_csv(OUT_CSV, index=False)
    print(f"Summary gespeichert in {OUT_CSV}")


if __name__ == "__main__":
    main()
