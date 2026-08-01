# AlphaZero 训练模块

公开使用与游玩说明见仓库根目录的 `README.md`。

本目录包含模型推理、MCTS、战术/VCF 求解、自我对弈训练、外部引擎适配评测和回归测试。仓库只跟踪当前已批准的 `latest.pt`；其他 checkpoint、回放、数据集和评测报告均为本地生成物。

当前桌面发布版本为 **Gargantua V3.2 R7**。`R7` 使用联盟自我博弈，同时保留Round 6高后悔、战术和白棋防守样本；候选通过独立Rapfi盲测和1024组交换色直接竞技场后才获准部署。

常用验证命令：

```powershell
python -m unittest discover -s alphazero_training -p "test_*.py"
python .\alphazero_training\verify_play_integration.py
```

## 标准化训练日志与运行控制

三个训练入口 `train_alphazero.py`、`train_v3_selfplay.py` 和
`train_v3_supervised.py` 默认启用统一训练审计。每次运行都会在
`alphazero_training/training_logs/<run_id>/` 保存：

- `manifest.json`：命令、完整配置、主机、Python、状态、起止时间和产物清单；
- `events.jsonl`：阶段、每步/每轮指标、验证、checkpoint 与控制事件，事件间使用
  SHA256 哈希链，修改历史记录会被校验器发现；
- `console.log`：完整标准输出和错误输出；
- `control.json`：可请求暂停、继续或在安全边界停止训练。

训练日志默认开启。可用 `--audit-root`、`--audit-run-id`、`--audit-mode` 和
`--audit-metric-every` 控制位置、运行编号、详细级别和指标频率。只有明确传入
`--audit-mode off` 才会关闭；正式训练不应关闭。也可用环境变量
`GOMOKU_TRAINING_LOG_DIR`、`GOMOKU_TRAINING_AUDIT` 和
`GOMOKU_TRAINING_AUDIT_EVERY` 设置未来训练的统一默认值。

```bash
# 查看最近训练
python -m alphazero_training.training_audit list

# 查看一轮训练的清单
python -m alphazero_training.training_audit status RUN_ID

# 在下一个安全边界暂停、继续或停止
python -m alphazero_training.training_audit control RUN_ID pause --reason "operator review"
python -m alphazero_training.training_audit control RUN_ID resume
python -m alphazero_training.training_audit control RUN_ID stop --reason "manual stop"

# 校验事件哈希链；加 --artifacts 会重新计算所有登记产物的 SHA256
python -m alphazero_training.training_audit verify RUN_ID --artifacts
```

暂停与停止均由训练循环主动读取 `control.json`，不会在写 checkpoint 的过程中粗暴
终止进程。日志目录属于本地/云端训练产物，默认不提交到 Git；精简后的验收报告仍可
按发布流程单独纳入版本控制。

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
