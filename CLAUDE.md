# workflow

## 1 workflow target
**最终目标**：在目标项目内完成环境搭建、在 mini 数据集上的 1-epoch 验证与所有 bugfix，最终输出可运行的验证记录与全量训练指导文档。

## 2. target project(current dir) workflow stages:

### target project workflow stage 1 — 环境构建与 mini 数据集准备
* **输入**：`venv.md`、`README.md`、`requirements-cpu.txt`、`requirements-gpu.txt`。
* **DO WHAT**：
    >执行 `python -m venv debug-cpu`，`pip install -r requirements-cpu.txt` 或者执行 `python -m venv debug-gpu`，`pip install -r requirements-gpu.txt`（要求用户二选一）；
    >Dataset 智能体按`venv.md`中规范抽样构建 mini数据集 并生成 `data/mini/`（样本数据、样本数、样本类别和分布等），创建notebook做可视化。
* **输出**：`env_check_report.md`（Python、包版本、可用 GPU 信息）、`data/mini/`（mini 数据集）与 `mini_dataset.md`（mini数据集使用说明和指标）。
* **WHO**：Ops 智能体 + Dataset 智能体
* **满足条件**：`env_check_report.md` 包含 `python_version`、`packages_installed`等环境安装和配置报告；mini 数据可被scripts目录下的notebook可视化。

### target project workflow stage 2 — 1-epoch 验证与自动 bugfix
* **输入**：`scripts/train.py`, `data/mini/`, `DEBUG_CODE.md`。
* **DO WHAT**：
    > 根据`DEBUG_CODE.md`规范,执行`scripts/train.py`, 1-epoch 快速训练（`--fast_dev_run` 或 `--epochs 1`）；
    > Coder智能体自动采集错误信息和修复bug，生成（或更新）bugfix报告；并再次测试直至通过。
* **输出**：`bugfix_report.md`（问题清单、bug摘要、bug修复日志、测试结果）、最终训练输出checkpoint模型文件。
* **WHO**：Coder智能体。
* **满足条件**：1-epoch 运行成功，`bugfix_report.md` 中所有“关键错误”标记为已修复。

### target project workflow stage 3 — 全量训练策略输出
* **输入**：`bugfix_report.md`、技术选型文档-`tech.md`。
* **DO WHAT**：根据`tech.md`生成全量训练指导文档 `full_train_guidance.md`，包括推荐的 batch\_size、learning\_rate schedule、epoch 数、checkpoint 策略、数据增强策略、分布式训练建议与监控点、各种性能配置的训练脚本示例。
* **输出**：全量训练指导文档-`full_train_guidance.md`（可直接用于 GPU 分布式训练环境）。
* **WHO**：Coder 智能体。
* **满足条件**：全量训练指导文档-`full_train_guidance.md` 包含明确的参数范围与部署步骤，且与 `tech.md` 中的资源估计一致。

# collaborative multi-agent systems
- **requirements-agent**，需求变更（change request/requirement update）
- **coder-agent**，编写代码
- **reviewer-agent**，代码完整性检查，代码质量评审，对齐需求（requirements.md）和需求变更（requirement_update_number.md）
- **ops-agent**,创建python venv环境；创建docker环境（CPU或者GPU）
- **tester-agent**，代码测试、功能测试和模块测试 
- **dataset-agent**，找到ML算法对应的最适合的开源数据集；为1-epoch训练抽取小数据集；数据采样和可视化

# AI action rules
- **卓越工作原则**， 所有代码都要检查一遍，不可以遗漏功能，不可以偷懒，绝对不能省略代码不写，必须要完整代码
- **绝不假设目标项目已存在**，创建前必须检查
- **严格遵守'.md'中规范要求**，按规范创建目录结构，按规范编写入argparse等
- **调试优先**：GPU环境验证代码正确性后再CPU部署
- **框架双栈**：同时支持PyTorch和PaddlePaddle

# tools
- **日志规范**，打印tool use 的详细信息