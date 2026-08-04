# Round7 champion acceptance

- Model version: `gomoku-v3-round7-20260801`
- Champion SHA256: `5e25cd5731084f1ca4e2eee9c7b20b2ea20f2719d4c51923a30f39567efcd49c`
- Prior champion SHA256: `04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894`
- Training: 50 league self-play iterations, 800 games, 25,511 replay positions, plus retained Round6 expert/tactical/white-defense quotas.
- Static gate: raw tactics 47/48; deployed tactics 48/48; white defense 16/18; safe probability mass 0.7922258036.
- User move-14 diagnostic: recorded but not passed; this was a mandatory diagnostic and not an independent veto under the approved acceptance policy.
- Independent Rapfi blind test: 1,024 paired openings; candidate score 0.255371 vs parent 0.228027; gains 145, losses 89; exact two-sided p = 0.000304964; no color regression.
- Direct candidate-versus-champion arena: 1,024 paired color-swapped openings (2,048 games); candidate score 0.542480; gains 136, losses 49; exact two-sided p = 1.16655e-10; both color scores improved.
- Post-deployment replay: raw tactics 47/48; deployed tactics 48/48; white defense 16/18; safe probability mass 0.7922258036.

All compact evidence files in this directory are retained with SHA256 sidecars. Raw game logs, replay data, and large audit streams remain on the cloud host and are intentionally excluded from Git.
