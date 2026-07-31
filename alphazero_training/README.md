# AlphaZero 训练模块

公开使用与游玩说明见仓库根目录的 `README.md`。

本目录包含模型推理、MCTS、战术/VCF 求解、自我对弈训练、外部引擎适配评测和回归测试。仓库只跟踪当前已批准的 `latest.pt`；其他 checkpoint、回放、数据集和评测报告均为本地生成物。

常用验证命令：

```powershell
python -m unittest discover -s alphazero_training -p "test_*.py"
python .\alphazero_training\verify_play_integration.py
```

## Rapfi 教师蒸馏与标准纠错流程

`rapfi_adapter.py` 通过 Piskvork 协议驱动 Rapfi。`rapfi_distill.py` 对每个随机开局
交换黑白各下一盘，并在每个非开局局面询问 Rapfi：既保留学生实际着法，也保留
教师建议和二者是否分歧。所有完整棋局写入带 SHA256 的 JSON，教师策略写入可由
`train_v3_supervised.py` 直接读取的 NPZ；学生败局同时写入指定的 `--ai-loss-dir`。

Linux/云端生成数据示例：

```bash
python -m alphazero_training.rapfi_distill \
  --checkpoint alphazero_training/latest.pt \
  --engine /path/to/rapfi-runtime/pbrain-rapfi \
  --report rapfi_distillation/train/games.json \
  --dataset rapfi_distillation/train/rapfi_policy.npz \
  --ai-loss-dir rapfi_distillation/train/pending_training/ai_losses \
  --pairs 128 --opening-plies 4 --simulations 64 --workers 8 \
  --timeout-turn-ms 300 --max-nodes 100000 --engine-threads 4
```

纠错和晋级必须按以下顺序执行：

1. 以 `pair_index` 分组训练，禁止同一交换色开局跨入训练集和验证集。
2. 用 `v3_candidate_gate freeze` 冻结候选，保留当前 `latest.pt` 不变。
3. 独立检查 48 个合法战术题和 18 个白方安全题；任一计数下降即淘汰。
4. 用全新随机种子分别评测父模型和候选，并运行：

```bash
python -m alphazero_training.compare_rapfi_reports \
  --parent-report eval/parent/games.json \
  --candidate-report eval/candidate/games.json \
  --output eval/comparison.json
```

比较器会验证报告哈希、开局、教师权重和搜索预算完全一致，再给出逐开局增益/退步、
黑白分项和精确配对符号检验。Rapfi 得分更高但 `p >= 0.05` 时只允许继续扩大独立
评测，不自动覆盖冠军。

默认桌面搜索预算为 256 MCTS，可通过环境变量 `GOMOKU_MCTS_SIMULATIONS` 临时覆盖。

桌面端每盘棋会按 V3 自我博弈的核心数组格式写入 `play_logs/all_games/`；AI 输掉的
完整棋局还会单独复制到 `play_logs/pending_training/ai_losses/`，作为人工审核后再训练
的待训练谱库。中途重开或关闭的棋局也保留每一步，但其未知胜负的 value weight 为 0。
