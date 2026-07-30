# AlphaZero 训练模块

公开使用与游玩说明见仓库根目录的 `README.md`。

本目录包含模型推理、MCTS、战术/VCF 求解、自我对弈训练、外部引擎适配评测和回归测试。仓库只跟踪当前已批准的 `latest.pt`；其他 checkpoint、回放、数据集和评测报告均为本地生成物。

常用验证命令：

```powershell
python -m unittest discover -s alphazero_training -p "test_*.py"
python .\alphazero_training\verify_play_integration.py
```

默认桌面搜索预算为 256 MCTS，可通过环境变量 `GOMOKU_MCTS_SIMULATIONS` 临时覆盖。
