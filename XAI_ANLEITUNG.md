# Explainability-Skripte — Anleitung

Diese Skripte liegen bereits in den vier Experiment-Ordnern und sind commit-fertig.
Du musst sie nicht mehr anpassen, nur pushen und auf dem Cluster starten.

## 1. Push (hier, auf diesem Rechner)

```bash
cd "/Users/janschlich/Desktop/Uni/6. Semester/Bachelor Thesis/Abgabe/SemEval_Task2_Jan_Schlich"
git add 1.2_RegressionBaseline/group_error_analysis.py 1.2_RegressionBaseline/group_error_analysis.sh \
        2.1_Temporal/group_error_analysis.py 2.1_Temporal/group_error_analysis.sh \
        2.1_Temporal/counterfactual_ablation.py 2.1_Temporal/counterfactual_ablation.sh \
        2.1_Temporal/attention_analysis.py 2.1_Temporal/attention_analysis.sh \
        2.2_UserID/group_error_analysis.py 2.2_UserID/group_error_analysis.sh \
        2.2_UserID/counterfactual_ablation.py 2.2_UserID/counterfactual_ablation.sh \
        2.2_UserID/attention_analysis.py 2.2_UserID/attention_analysis.sh \
        3_Regression_Combined/group_error_analysis.py 3_Regression_Combined/group_error_analysis.sh \
        3_Regression_Combined/counterfactual_ablation.py 3_Regression_Combined/counterfactual_ablation.sh \
        3_Regression_Combined/attention_analysis.py 3_Regression_Combined/attention_analysis.sh \
        XAI_ANLEITUNG.md
git commit -m "add: explainability scripts + sbatch jobs (group error, counterfactual ablation, attention)"
git push
```

## 2. Auf dem Cluster: pullen und venv aktivieren

```bash
git pull
source /home/jaschlic/venv/bin/activate
pip install matplotlib --upgrade   # falls noch nicht installiert
```

## 3. Reihenfolge pro Ordner

Für jeden Ordner gilt: erst `group_error_analysis.sh` (braucht nur `predictions.csv`,
läuft schnell durch), dann `counterfactual_ablation.sh` und `attention_analysis.sh`
(beide brauchen den trainierten Checkpoint in `../models/` und laufen mit GPU, deshalb
`sbatch` statt direkt `python`, genau wie bei `run.sh`/`run_ablation.sh`).

```bash
# 1.2 — nur Kontrollgruppen-Analyse (kein Temporal-/User-ID-Mechanismus vorhanden)
cd 1.2_RegressionBaseline
sbatch group_error_analysis.sh
cd ..

# 2.1 — Temporal-Modell
cd 2.1_Temporal
sbatch group_error_analysis.sh
sbatch counterfactual_ablation.sh
sbatch attention_analysis.sh
cd ..

# 2.2 — User-ID-Modell
cd 2.2_UserID
sbatch group_error_analysis.sh
sbatch counterfactual_ablation.sh
sbatch attention_analysis.sh
cd ..

# 3 — kombiniertes Modell (Temporal + User-ID)
cd 3_Regression_Combined
sbatch group_error_analysis.sh
sbatch counterfactual_ablation.sh
sbatch attention_analysis.sh
cd ..
```

Mit `squeue -u jaschlic` (oder euren Username) siehst du den Job-Status. Jeder Job
schreibt sein eigenes Log (`group_error_log.txt`/`_error.txt`,
`counterfactual_log.txt`/`_error.txt`, `attention_log.txt`/`_error.txt`) im jeweiligen
Ordner — kollidiert nicht mit den bestehenden `run_log.txt`/`ablation_log.txt`.

`counterfactual_ablation.sh` und `attention_analysis.sh` können unabhängig voneinander
und parallel zu `group_error_analysis.sh` laufen; nur muss `run.py` für den jeweiligen
Ordner vorher durchgelaufen sein (Checkpoint + `predictions.csv` müssen existieren).

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
