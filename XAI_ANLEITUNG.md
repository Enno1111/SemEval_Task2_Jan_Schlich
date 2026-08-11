# Explainability-Skripte — Anleitung

Diese Skripte liegen bereits in den vier Experiment-Ordnern und sind commit-fertig.
Du musst sie nicht mehr anpassen, nur pushen und auf dem Cluster starten.

## 1. Push (hier, auf diesem Rechner)

```bash
cd "/Users/janschlich/Desktop/Uni/6. Semester/Bachelor Thesis/Abgabe/SemEval_Task2_Jan_Schlich"
git add 1.2_RegressionBaseline/group_error_analysis.py \
        2.1_Temporal/group_error_analysis.py \
        2.1_Temporal/counterfactual_ablation.py \
        2.1_Temporal/attention_analysis.py \
        2.2_UserID/group_error_analysis.py \
        2.2_UserID/counterfactual_ablation.py \
        2.2_UserID/attention_analysis.py \
        3_Regression_Combined/group_error_analysis.py \
        3_Regression_Combined/counterfactual_ablation.py \
        3_Regression_Combined/attention_analysis.py \
        XAI_ANLEITUNG.md
git commit -m "add: explainability scripts (group error, counterfactual ablation, attention)"
git push
```

## 2. Auf dem Cluster: pullen und venv aktivieren

```bash
git pull
source .venv/bin/activate   # oder wie euer venv dort heisst
pip install matplotlib --upgrade   # falls noch nicht installiert
```

## 3. Reihenfolge pro Ordner

Für jeden Ordner gilt: erst `group_error_analysis.py` (braucht nur `predictions.csv`,
läuft sofort, kein GPU nötig), dann `counterfactual_ablation.py` und
`attention_analysis.py` (beide brauchen den trainierten Checkpoint in `../models/`,
laufen am besten mit GPU).

```bash
# 1.2 — nur Kontrollgruppen-Analyse (kein Temporal-/User-ID-Mechanismus vorhanden)
cd 1.2_RegressionBaseline
python group_error_analysis.py
cd ..

# 2.1 — Temporal-Modell
cd 2.1_Temporal
python group_error_analysis.py
python counterfactual_ablation.py
python attention_analysis.py
cd ..

# 2.2 — User-ID-Modell
cd 2.2_UserID
python group_error_analysis.py
python counterfactual_ablation.py
python attention_analysis.py
cd ..

# 3 — kombiniertes Modell (Temporal + User-ID)
cd 3_Regression_Combined
python group_error_analysis.py
python counterfactual_ablation.py
python attention_analysis.py
cd ..
```

Jedes Skript schreibt seine Ergebnisse in einen neuen Unterordner
`<Experiment>/explainability_out/`:

- `group_error_analysis.py` → `group_error_summary.csv`, `residuals_valence.png`, `residuals_arousal.png`
- `counterfactual_ablation.py` → `counterfactual_ablation_full.csv` (pro Zeile), `counterfactual_ablation_summary.csv` (pro Gruppe)
- `attention_analysis.py` → `attention_mass_by_group.csv` (pro Sample), `attention_mass_summary.csv` (pro Gruppe)

## 4. Danach zurück pushen

```bash
git add */explainability_out
git commit -m "results: explainability analysis"
git push
```

Sag mir Bescheid, sobald das durch ist — dann schauen wir uns die Ergebnisse zusammen an
und schreiben daraus das Explainability-Kapitel.

## Offener Punkt (nicht blockierend für XAI)

`3_Regression_Combined/model.py` nutzt noch `MIN_USER_TEXTS = 15` und
`USER_ID_LENGTH = 2`. Die neue, deterministische Ablation in `2.2_UserID` hat aber
`MIN_USER_TEXTS = 20` / `USER_ID_LENGTH = 5` als Sieger ermittelt. Laut Methodology-Text
(Abschnitt 4.4) sollen für das kombinierte Modell "beide besten Konfigurationen" aus
4.2 und 4.3 übernommen werden — das würde bedeuten, Combined müsste ebenfalls auf 20/5
umgestellt und neu trainiert werden. Das betrifft nur die Zahlen in Kapitel 4/Results,
nicht die Explainability-Skripte selbst — die funktionieren mit der aktuellen
Konfiguration genauso. Am besten vor dem finalen Ergebniskapitel klären, nicht mehr
jetzt vor der XAI-Analyse.
