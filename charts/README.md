# Gargantua Training and Evaluation Charts

This folder contains presentation-ready, bilingual PNG charts generated from
the archived Gargantua training logs and held-out evaluations.

## Files

- `01_supervised_warmup_loss.png` — loss components during 800 supervised warm-up steps.
- `02_selfplay_training_loss.png` — loss components across 100 self-play iterations.
- `03_selfplay_data_generation.png` — replay-buffer growth and new self-play positions.
- `04_tactical_evaluation_over_training.png` — raw-network tactical performance every five iterations.
- `05_white_defense_evaluation_over_training.png` — held-out white-defense performance every five iterations.
- `06_approved_model_results.png` — final approved Gargantua V2 validation summary.
- `00_all_charts_preview.png` — contact sheet for quick browsing.
- `data/` — CSV snapshots and provenance notes used to draw the charts.

## Important interpretation

Training loss measures how well the network fits its training targets; it is
not direct evidence of playing strength. The 159 integration checks measure
software correctness and tactical routing, not a 159-game win record. The
100-iteration experimental run did not outperform the approved Gargantua V2
checkpoint on the held-out top-1 gates, so it was not deployed.

## Regenerate

Run the following command from the `Final Project` directory:

```powershell
python .\charts\generate_training_charts.py --archive "PATH_TO_V3G_ARCHIVE"
```
