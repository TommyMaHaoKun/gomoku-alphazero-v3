# Gargantua — Gomoku AI

A playable 19×19 Gomoku program built with Pygame and an AlphaZero-style AI named **Gargantua**. The player can choose black or white, while Gargantua combines a policy-value neural network, Monte Carlo Tree Search (MCTS), and exact tactical checks.

The complete final-project report is available in [`summary.md`](summary.md).

## Features

- Standard 19×19 Gomoku board with nine star points
- Human can choose black (first) or white (second)
- Move numbers are displayed on every stone
- The AI's latest move is marked with a square outline
- Immediate win and mandatory-block detection
- Policy-value network guided MCTS
- Automatic CPU/GPU inference selection
- Current model and MCTS configuration shown in the game window
- Automatic move-by-move game logging in AlphaZero replay format
- Separate pending-training library for games lost by the AI

## Libraries

The playable program uses:

| Library | Version | Purpose |
| --- | --- | --- |
| Python | 3.11 tested | Main programming language |
| Pygame | 2.6.1 | Window, board rendering, mouse and keyboard input |
| PyTorch | 2.8.0 | Policy-value neural network training and inference |
| NumPy | 2.0 or newer | Board tensors, replay data, sampling and numerical operations |

The training tools also use Python standard-library modules such as `multiprocessing`, `threading`, `queue`, `pathlib`, `dataclasses`, `json`, and `logging`.

## Installation and Play

Install the required packages:

```powershell
python -m pip install -r alphazero_training/requirements_play.txt
```

Start the game:

```powershell
python "Gomoku AI player V1.0.py"
```

Controls:

- Click **B: BLACK (FIRST)** or press `B` to play black.
- Click **W: WHITE (SECOND)** or press `W` to play white.
- Press `R` or `Enter` to restart.
- Press `C` to choose a different color.
- Press `S` to show or hide heuristic scores.

The default desktop search budget is 256 MCTS simulations per move. It can be changed without editing the source code:

```powershell
$env:GOMOKU_MCTS_SIMULATIONS = "128"
python "Gomoku AI player V1.0.py"
```

## Game Logs and Pending Training Games

Every desktop game is archived under `alphazero_training/play_logs/all_games/`.
Each game has a readable JSON move list and a compressed NPZ replay containing
the same five core arrays as V3 self-play: `states`, `policies`, `values`,
`policy_weights`, and `value_weights`. Human and AI actions use one-hot played-
move policy targets. If a game is restarted, its known moves are retained while
the unknown result is masked with zero value weights.

When the human wins, the matching JSON and NPZ files are also copied to
`alphazero_training/play_logs/pending_training/ai_losses/`. This is a review
queue for later training; playing the game does not automatically change or
retrain the deployed model.

## AI Method

This project uses an AlphaZero-style hybrid rather than a neural network alone. It combines reinforcement learning from self-play, supervised tactical warm-start experiments, MCTS, and deterministic tactical safeguards.

### Policy-value network

The deployed network receives a four-plane board representation. Its residual tower uses 96 channels and 8 residual blocks, then produces:

- a **policy output** over all 361 board locations;
- a **value output** estimating the current player's expected result.

### MCTS and tactical search

MCTS uses the policy output as a move prior and the value output to evaluate leaf positions. PUCT balances promising moves with exploration. During desktop play, an exact tactical layer runs through the same search interface and gives priority to immediate wins, mandatory blocks, and bounded forced-win sequences before the final MCTS decision.

### Candidate reduction

Search is concentrated around relevant stones instead of expanding every empty point equally. The deployed configuration uses a candidate radius of 2 and blends neural priors with a small handcrafted positional prior.

## Training Process

Training was performed on an NVIDIA CUDA GPU using repeated self-play and optimization cycles:

1. **Self-play generation** — the current champion played games against itself with MCTS. Early moves used temperature sampling and Dirichlet noise to create varied positions.
2. **Replay storage** — each position stored its board state, MCTS visit distribution, and final game result. The replay buffer held up to 500,000 positions.
3. **Network update** — batches of 512 positions trained the policy and value heads with AdamW. The total loss was the policy cross-entropy plus value mean-squared error.
4. **Candidate evaluation** — every five iterations, a candidate played paired games from shared randomized openings, with colors exchanged for fairness.
5. **Promotion or rollback** — a candidate was promoted only after passing the evaluation threshold; otherwise training weights were reset to the accepted champion.
6. **Checkpointing** — model, optimizer, iteration, configuration, and replay metadata were saved so training could resume safely.

The base training program can be started or continued with:

```powershell
python -m alphazero_training.train_alphazero `
  --output run `
  --init-checkpoint alphazero_training/latest.pt `
  --max-iterations 450
```

Training requires a CUDA-capable GPU. If `run/latest.pt` already exists, the program resumes from that checkpoint automatically.

The repository also contains experimental supervised tactical warm-start and reinforced self-play pipelines. Experimental checkpoints are kept separate from the playable model and are not deployed unless they pass the non-regression checks.

## Current Model and Results

The game currently loads the approved checkpoint below:

| Item | Result |
| --- | --- |
| Displayed model | Gargantua V3.2 R7 |
| Board size | 19×19 |
| Training release | Round 7 league self-play with Round 6 regret, tactical, and white-defense retention |
| Desktop search | 256 MCTS simulations per move |
| Held-out legal tactical positions | 47 / 48 correct from the raw network |
| White-defense safe top move set | 16 / 18 correct |
| Independent Rapfi blind test | Candidate score 0.2554 vs 0.2280, `p = 0.000305` vs previous champion |
| Direct color-swapped arena | 54.25% over 2,048 games, `p = 1.17e-10` |
| Pygame integration regression suite | 159 tactical decisions passed |

The interface-level tactical guard also checks immediate wins and mandatory blocks before accepting a search result. These results show that the program can conduct practical games as either color, but they should not be interpreted as proof that the model is unbeatable.

Checkpoint SHA256:

```text
5e25cd5731084f1ca4e2eee9c7b20b2ea20f2719d4c51923a30f39567efcd49c
```

## Validation

Run the complete unit-test collection:

```powershell
python -m unittest discover -s alphazero_training -p "test_*.py"
```

Run the desktop integration regression suite:

```powershell
python .\alphazero_training\verify_play_integration.py
```

## Project Structure

```text
Gomoku AI player V1.0.py       Main Pygame human-vs-AI program
Gomoku V1.0.py                 Lightweight original board program
alphazero_training/latest.pt   Approved playable checkpoint
alphazero_training/play_agent.py
                               Model-loading and move-selection adapter
alphazero_training/train_alphazero.py
                               Self-play training loop
alphazero_training/v3_search.py
                               Tactical routing and MCTS search
alphazero_training/test_*.py   Training, search and integration tests
```
