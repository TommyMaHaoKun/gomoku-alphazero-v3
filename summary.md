# Gargantua: Gomoku AI Final Project Summary

## Abstract

This project developed an executable 19×19 Gomoku game in Python with a trained computer opponent named **Gargantua**. The main objective was to move beyond a purely rule-based opponent and create an AI that could make strategically meaningful decisions as either black or white. The final system combines a Pygame interface, a PyTorch policy-value neural network, Monte Carlo Tree Search (MCTS), and deterministic tactical safeguards. The neural network was trained through AlphaZero-style self-play reinforcement learning, while a replay buffer, paired evaluation games, checkpointing, and rollback rules were used to make training more stable. The approved playable checkpoint is Gargantua V2 at iteration 174. It contains a 96-channel, 8-residual-block network with approximately 1.64 million trainable parameters. On the recorded validation suites, its raw network selected the correct move in 47 of 48 held-out legal tactical positions, selected a safe white move in 16 of 18 white-defense positions, and passed 159 Pygame integration decisions. The completed project includes a playable interface, selectable player color, move numbering, AI-move highlighting, reproducible tests, and a public GitHub repository containing the executable code and approved model.

## 1. Problem and Motivation

Gomoku is easy to describe but difficult to play well. Two players alternate placing stones on a grid, and the first player to form a continuous line of five stones wins. A basic program can enforce these rules with only a small amount of code. Building a useful opponent is much harder because the number of possible positions grows rapidly, and a good move may need to balance immediate defense, long-term attack, and several interacting threats.

The original version of this project used handcrafted pattern scores. It counted connected stones, open ends, and several common shapes, then selected a move with a high manually assigned score. This approach was useful as a starting point because it was fast and understandable. However, it had three important weaknesses. First, the quality of the opponent depended directly on patterns written by the programmer. Second, a local pattern score could miss a multi-step tactical consequence. Third, the same heuristic did not always perform equally well as black and white.

The project therefore explored whether a learned policy-value model and tree search could produce a stronger and more flexible opponent. Gomoku was also a suitable educational project because it connects several course topics in one system: Python programming, arrays, neural networks, reinforcement learning, search algorithms, multiprocessing, data validation, and graphical user-interface development.

The final goal was not simply to obtain a low training loss. The program had to run reliably, respond to mouse and keyboard input, follow the game rules, block immediate losses, play as either color, show the model being used, and load the same approved checkpoint every time. Model quality therefore had to be evaluated together with software correctness.

## 2. Project Objectives and Deliverables

The project was organized around five concrete objectives:

1. **Create a complete playable application.** The user should be able to start the program, choose black or white, place legal moves, restart, and finish a game.
2. **Integrate a trained model.** The computer opponent should use a policy-value neural network rather than depend entirely on handcrafted scores.
3. **Combine learning and search.** Neural predictions should guide MCTS, while exact tactical checks should prevent simple missed wins or blocks.
4. **Build a repeatable training workflow.** Self-play data, replay files, checkpoints, logs, and evaluation results should be recoverable after interruption.
5. **Evaluate conservatively.** A newly trained checkpoint should replace the playable model only when it passes tactical, integration, and color-balanced non-regression checks.

The final deliverables are the Pygame application, the approved `latest.pt` model, the training and evaluation modules, automated tests, documentation, and the public GitHub repository. Large replay buffers and experimental training archives are retained locally rather than published as part of the executable repository.

## 3. System Design

### 3.1 Pygame interface

The graphical interface is implemented with Pygame. It renders the 19×19 board, nine star points, black and white stones, control buttons, status text, and model information. The user may choose black or white. Each stone displays its move number, and a square outline marks only the AI's latest move. The right side of the window reports the loaded model version and the active MCTS budget.

The interface is kept separate from the neural-network implementation. The Pygame program stores the visible board and sends a numerical representation of the current position to `AlphaZeroGomokuAgent`. The agent returns an `(x, y)` coordinate, which the interface checks before updating the board. AI search runs outside the rendering loop so that the window remains responsive while the model is thinking.

### 3.2 Board representation

The model receives four 19×19 input planes:

1. stones belonging to the player whose turn it is;
2. stones belonging to the opponent;
3. the location of the most recent move;
4. a constant plane indicating whether the current player is black.

Separating the position into planes lets convolutional layers process spatial relationships without assigning arbitrary numerical distances between black stones, white stones, and empty intersections.

### 3.3 Policy-value neural network

The approved model uses 96 internal feature channels and 8 residual blocks. A channel is a 19×19 feature map. The model learns these maps automatically, so different channels can respond to different spatial patterns such as connected stones, open lines, defensive shapes, or intersections relevant to multiple threats.

Each residual block contains two 3×3 convolutional layers with batch normalization. Its output is added to the block's original input before the activation function. If the transformation learned by a block is written as `F(x)`, the block returns:

```text
output = x + F(x)
```

This skip connection helps information and gradients move through a deeper network. Eight residual blocks therefore do not mean only eight individual hidden layers: each block contains multiple layers, and the network also has an input stem and separate output heads.

The policy head returns one logit for each of the 361 board locations. The value head returns a number between -1 and 1 that estimates the expected game result from the current player's perspective. The complete model has approximately 1,641,586 trainable parameters.

### 3.4 Search and tactical safeguards

The project uses MCTS to convert network predictions into a move. The policy output gives prior probabilities for candidate moves, while the value output estimates leaf positions. During tree selection, PUCT combines the current average value of a move with an exploration term. This encourages the search to revisit strong moves while still investigating moves that have high neural-network priors but fewer visits.

The search is restricted to relevant candidate positions around existing stones, using a radius of two in the deployed configuration. A small handcrafted pattern prior is blended with the neural prior, but the neural value and tree statistics still control the search decision.

The final system is intentionally hybrid. Before accepting a normal search result, the tactical layer checks immediate wins, mandatory blocks, and bounded forced-win sequences. This design addresses a practical weakness of small policy-value networks: even a generally strong model may assign too little probability to one tactically required move. The desktop application uses 256 MCTS simulations per move by default and automatically uses CUDA when a supported GPU is available.

## 4. Training Method

### 4.1 Self-play reinforcement learning

Training data was generated by allowing the current champion to play against itself. At every position, MCTS produced a visit-count distribution over legal moves. This distribution became the policy target. After the game ended, the final result became the value target for every stored position, with the sign adjusted according to the player to move.

Early moves used temperature sampling so that training games did not always follow one deterministic opening. Dirichlet noise was added at the search root during self-play to encourage additional exploration. These mechanisms increased position diversity and reduced the chance that the model would learn only a narrow group of openings.

Multiple games were processed in parallel, and neural-network evaluations were batched on the GPU. The approved checkpoint configuration used up to 64 parallel games, 128 self-play games per iteration, and 192 MCTS simulations for training and arena evaluation.

### 4.2 Replay buffer and optimization

Each self-play position stored three items: the four-plane board state, the MCTS policy target, and the final value target. New positions were added to a replay buffer with a maximum capacity of 500,000. Sampling from a replay buffer mixes recent games with older positions and reduces the correlation between consecutive training examples.

The network was optimized with AdamW using batches of 512 positions. Its objective combined policy and value losses:

```text
total loss = policy cross-entropy + value mean-squared error
```

The policy loss encouraged the network distribution to match the stronger search distribution. The value loss encouraged the predicted outcome to match the actual self-play result. The base configuration performed 300 gradient steps during each training iteration.

### 4.3 Candidate evaluation and promotion

Training loss alone was not treated as evidence of playing strength. At regular intervals, the newly trained candidate was evaluated against the accepted champion. Games used shared randomized openings, and colors were exchanged so that both models faced the same opening conditions as black and white. A candidate was promoted only if it reached the configured score threshold. If it failed, its weights were reset to the accepted model before training continued.

Later experiments added tactical supervision, white-defense examples, harder negative positions, and stronger self-play searches. These experimental checkpoints were kept separate from `latest.pt`. A later iteration number was not automatically considered better; deployment required non-regression on held-out tactics and both colors. This rule prevented an experimental model from replacing a more reliable playable checkpoint.

### 4.4 Checkpointing and recovery

Each checkpoint stored the training model, accepted model, optimizer state, configuration, iteration, and replay size. Replay chunks and logs were saved separately. Checkpoints were written atomically so that an interrupted write would not destroy the previous usable model. When an output directory already contained `latest.pt`, the trainer loaded it and resumed from the next iteration.

This workflow was important because self-play was much slower than a single neural-network update. Most training time was spent exploring full games and building MCTS trees, so preserving completed self-play data saved both time and GPU cost.

## 5. Implementation and Libraries

The project was implemented in Python and used three main external libraries:

- **Pygame 2.6.1** for rendering, input events, buttons, status messages, and the playable desktop loop.
- **NumPy 2.0 or newer** for board arrays, encoded states, replay storage, random sampling, and numerical operations.
- **PyTorch 2.8.0** for the residual policy-value network, optimization, checkpoint loading, batched inference, and CUDA acceleration.

Python standard-library modules provided multiprocessing, queues, threads, data classes, paths, JSON processing, logging, hashing, and atomic file operations. The training code required an NVIDIA CUDA GPU, while the final application could run on either CPU or GPU.

The program was divided into modules rather than placing training, search, and interface logic in one file. `train_alphazero.py` defines the game state, network, replay buffer, base MCTS, training loop, arena, and checkpoint format. `v3_search.py` and `tactical_solver.py` implement stronger search routing. `play_agent.py` provides a small adapter between a checkpoint and the Pygame board. This separation also made it possible to test search decisions without launching an interactive window.

## 6. Results and Evaluation

The deployed model is **Gargantua V2, iteration 174**. Gargantua is the name of the project's playable model; its architecture and training method remain AlphaZero-style. Its checkpoint records a full replay buffer of 500,000 positions and uses 96 channels with 8 residual blocks. The SHA256 hash of the published checkpoint is:

```text
ad2082e7d0223a42047cf0b349b0de17b7fe88c0145dea95c9f2bbe709c6c96e
```

Three forms of evaluation were used:

| Evaluation | Result |
| --- | ---: |
| Held-out legal tactical positions, raw network | 47 / 48 |
| White-defense safe top-move set | 16 / 18 |
| Pygame integration decisions | 159 passed |

The raw-network result measures whether the policy itself ranked a correct tactical move first, without counting a perfect result produced by an exact tactical override. This distinction matters because it separates learned behavior from behavior guaranteed by the surrounding solver. The white-defense set checks whether the top move belongs to a defined safe set in positions where defensive accuracy is important.

The integration suite tests the complete path used by the playable program. It covers immediate wins and blocks for both colors, the exact position associated with an earlier missed-block failure, move numbering, color selection, AI-move outlines, model labels, legal move conversion, and several regression cases. In the latest complete run, all 165 unit tests passed in addition to the 159 integration decisions.

The final application met the practical project objectives. It can play as black or white, makes legal moves, blocks immediate threats, identifies direct wins, preserves a responsive window during search, and reports the exact deployed model. The result is substantially more capable than the initial local-pattern program. However, the numerical results are test-set-specific and do not prove that the program is unbeatable.

## 7. Challenges and Limitations

The largest challenge was that additional training did not always improve the model. Self-play can reinforce weaknesses already present in the current policy, and aggregate training loss can decrease even when one important tactical position becomes worse. Black and white also have different strategic conditions because black moves first. A model can therefore improve its overall self-play objective while becoming less reliable as one color.

Search speed was another limitation. The policy-value network is small enough for desktop use, but 256 MCTS simulations still require noticeable time on a CPU. Increasing the simulation budget generally improves search at the cost of a slower response. GPU acceleration helps neural inference, but tree construction and tactical checks also use CPU work.

The tactical and white-defense evaluation sets are useful regression tests but remain small. A result such as 47/48 gives clear evidence about those positions, not every possible Gomoku situation. The self-play opening distribution also cannot represent the entire game tree. Finally, the current rules implement standard freestyle five-in-a-row and do not include every tournament opening or forbidden-move rule variation.

These limitations influenced the final deployment decision. The project retained an older approved model when later experimental checkpoints failed the non-regression requirements. This was preferable to publishing a newer checkpoint only because it had consumed more training time.

## 8. Future Work

Future development should focus on data quality and evaluation rather than only increasing the number of training iterations. A larger held-out tactical suite could include more defensive forks, edge positions, long forced sequences, and balanced black/white examples. Difficult failures from real games could be added to training through a controlled replay curriculum while remaining separate from final evaluation data.

The model could also be improved with a larger residual tower, more feature channels, or longer training, but any architecture change should be compared under the same openings and search budget. Distillation from a stronger search could improve the raw policy and reduce the number of simulations needed during desktop play. Performance engineering could batch multiple tree leaves more efficiently and move additional search operations to compiled code.

On the interface side, useful additions would include adjustable difficulty levels, saved game records, move undo, replay analysis, and visual explanations of why the AI selected a move. Rule variants could be implemented as separate game configurations rather than changing the existing freestyle behavior.

## 9. Conclusion

This project transformed a basic Pygame Gomoku program into a documented, tested, and executable AI application. The final design combines a residual policy-value network, self-play reinforcement learning, MCTS, and deterministic tactical protection. Just as importantly, it uses replay management, paired evaluation, checkpoint recovery, and conservative promotion rules to distinguish a completed training run from a genuinely better model.

The approved iteration-174 checkpoint and the final interface satisfy the main deliverables: the user can play either color, the program handles essential tactical situations, the model is clearly identified, and the repository includes the code and tests required to reproduce the application. The remaining weaknesses provide concrete directions for future work in balanced training data, larger independent evaluations, faster search, and more interpretable gameplay.
