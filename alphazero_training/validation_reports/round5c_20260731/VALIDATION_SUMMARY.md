# Round 5c champion validation

Validated on 2026-07-31 against the previously deployed champion using an independent 1,024-opening, 2,048-game paired Rapfi blind test.

- Candidate SHA256: `04a75eeef57d7221fd90df23c285d7c788d265cfbb29495b7391156f02939894`
- Previous champion SHA256: `ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e`
- Rapfi score: `0.1865234375` vs `0.1669921875` (`+0.01953125`)
- Paired openings: 168 gains, 128 losses, 728 unchanged
- Two-sided exact sign test: `p = 0.023238852227195444`
- Black score: `0.373046875` vs `0.333984375`; white score: `0.0` vs `0.0`
- Raw tactics: 47/48; deployed tactics: 48/48
- White-defense safety: 16/18, safe probability mass `0.7707617002141821`
- User loss-game move 14: 256-MCTS selected teacher action 180 `(9, 9)`; teacher raw-policy rank 1 with probability `0.4245220720767975`
- Blind reports: 1,024/1,024 pairs complete for both models, zero errors and zero truncated games
- Local Python 3.11 validation: 18 focused unit tests passed; 159 desktop integration decisions passed

All hashes embedded in `FINAL_DEPLOYMENT_GATE.json` were recomputed after download and matched the downloaded model and reports. The previous local champion is retained as an ignored backup under `alphazero_training/candidates/`.
