# Dissertation Handoff

## Project context

Romanian Master's dissertation on missing values in time series forecasting with RNN/LSTM/GRU.
Title: *Analiza, tratarea și efectul valorilor lipsă în predicția seriilor de timp cu tehnici de învățare automată.*
Author: Paul-Alin Tatar

Working directory: `/home/tatarpaul/Repos/AI_Project/Disertatie/`

---

## What was done in the previous session

### 1. Humanized all existing chapter files

Applied the `/humanizer` skill (swarm of 6 parallel agents) to remove AI writing patterns from all `.tex` chapters. The `.bib` file was left untouched.

| File | Status |
|------|--------|
| `Introducere.tex` | humanized |
| `StadiulActualAlCercetarii.tex` | humanized |
| `Metodologie.tex` | humanized |
| `ContributiiProprii.tex` | humanized |
| `Implementaresiexperimente.tex` | humanized |
| `AspecteTeoretice.tex` | humanized |
| `Bibliografie.bib` | untouched |

Patterns removed: em dashes, signposting openers, copula avoidance, significance inflation, passive voice, filler phrases, negative parallelisms, rule-of-three overuse, promotional language. Verified with `git diff` — no LaTeX commands touched, no dashes introduced.

### 2. Created Rezultate.tex (new chapter)

`Rezultate.tex` was written from scratch (~3600 words, ~15 pages when compiled) covering 6 experiments + a synthesis section:

1. Imputation strategy comparison — DSMTS visual (fill_zero MAPE 51% vs predictive 5.18%) + two result tables from CSV exports
2. Learning rate schedulers — 5 strategies compared (constant, step, exponential, cosine, warmup+cosine)
3. Federated learning — 10 users, 20 rounds
4. Federated learning — 10 users, 50 rounds (convergence comparison)
5. Parallel ensemble (independent topology) — 25 runs
6. Sequential ensemble (chain topology) — 5 stages
7. Synthesis section

Uses the `/humanizer` principles throughout — written naturally as a Master's student.

---

## Current file structure

All files are flat in `Disertatie/` (no subfolders):

```
Disertatie/
├── Thesis.tex                   ← main compile file
├── Introducere.tex
├── StadiulActualAlCercetarii.tex
├── AspecteTeoretice.tex
├── Metodologie.tex
├── ContributiiProprii.tex
├── Implementaresiexperimente.tex
├── Rezultate.tex                ← NEW, created this session
├── Bibliografie.bib
└── handoff.md                   ← this file
```

---

## IMPORTANT: Thesis.tex structure mismatch

`Thesis.tex` currently uses `\input{Capitole/...}` paths and expects a `Capitole/` subfolder that does **not exist**. The actual `.tex` files are all in the root `Disertatie/` folder.

**What needs to be fixed before compiling:**

1. Either update `Thesis.tex` to use flat paths:
   ```latex
   \input{Introducere}
   \input{StadiulActualAlCercetarii}
   \input{AspecteTeoretice}
   \input{Metodologie}
   \input{ContributiiProprii}
   \input{Implementaresiexperimente}
   \input{Rezultate}
   ```

2. Or create `Capitole/` and move files there (and rename to match exactly):
   - `Capitole/Rezultate și discuții.tex` ← note the spaces and diacritics in Thesis.tex
   - `Capitole/ContribuțiileProprii.tex` ← note the diacritic spelling differs

3. The chapter `ConcluziiSiDirectiiViitoatre.tex` is referenced in `Thesis.tex` but **does not exist yet** — needs to be written.

4. `Thesis.tex` already has `\graphicspath{{img/}}`, `\usepackage{float}`, `\usepackage{booktabs}`, `\usepackage{graphicx}` — no changes needed to the preamble for `Rezultate.tex`.

---

## Images for Rezultate.tex

Images live in `/home/tatarpaul/Repos/AI_Project/Screenshots/`. For Overleaf, upload to the `img/` folder.

**7 files — use original filename as-is:**

| Upload as | Source path |
|-----------|-------------|
| `Fill-0.png` | `Screenshots/Normal Forecast/Fill-0.png` |
| `PredictiveInputer.png` | `Screenshots/Normal Forecast/PredictiveInputer.png` |
| `lr_schedulers_test.png` | `Screenshots/lr_schedulers_test.png` |
| `training_comparison.png` | `Screenshots/training_comparison.png` |
| `StrategyComparison-10U-20R.png` | `Screenshots/Federated Learning/Experiment-10U-20R/StrategyComparison-10U-20R.png` |
| `PredictiveInputer-Real-Predicted.png` | `Screenshots/Federated Learning/Experiment-10U-20R/PredictiveInputer-Real-Predicted.png` |
| `StrategyComparison-10U-50R.png` | `Screenshots/Federated Learning/Experiment-10U-50R/StrategyComparison-10U-50R.png` |

**3 files — rename before uploading (naming conflicts in original):**

| Upload as | Source path |
|-----------|-------------|
| `parallel_rmse.png` | `Screenshots/Assembly Learning/Paralel Chain/image.png` |
| `parallel_diagnostics.png` | `Screenshots/Assembly Learning/Paralel Chain/image copy.png` |
| `sequential_diagnostics.png` | `Screenshots/Assembly Learning/Sequential Chain/image.png` |

---

## Result data (CSV files)

Two experiment result exports at `/home/tatarpaul/Repos/AI_Project/Results/`:
- `2026-04-03T17-36_export.csv` — Run 1 (likely lower missing % — fill_mean RMSE ~190)
- `2026-04-03T17-31_export.csv` — Run 2 (likely higher missing % — fill_mean RMSE ~350)

Both cover 4 imputation strategies × 3 models (12 rows each). These are used in `Rezultate.tex` Section 1, Tables 1 and 2.

---

## What still needs to be done

- [ ] Write `ConcluziiSiDirectiiViitoatre.tex` (Conclusions chapter — referenced in Thesis.tex but missing)
- [ ] Fix `Thesis.tex` input paths to match actual file locations/names
- [ ] Upload images to Overleaf `img/` folder (see table above)
- [ ] Test compile in Overleaf end-to-end
- [ ] Check bibliography: `Bibliografie.bib` is the source but Thesis.tex references `\bibliography{Bibliografia}` — filename mismatch (one has `e` at the end, one doesn't). Verify which is correct.

---

## Style notes for future writing

- Language: Romanian academic
- Style target: natural Master's student voice, not AI-generated
- Apply `/humanizer` skill to any new text before finalizing
- No em/en dashes anywhere in prose
- No signposting ("În primul rând... Apoi... În concluzie...")
- No "ceea ce demonstrează că", "este de menționat că", "În plus," chains
- Technical terms stay exact: RNN, LSTM, GRU, MCAR, MSE, MAE, RMSE, MAPE, ffill, fill_mean, window_mean, predictive_imputer
- All LaTeX commands (\cite, \ref, \label, math envs) preserved verbatim
