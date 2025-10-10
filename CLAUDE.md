# workflow
> 包含2个工作流，只能跟随其中一个：
- **bugfix workflow**: 代码修复工作流要求完成环境搭建；测试训练代码并完成1-epoch验证
- **requirement change workflow**：需求变更工作流要求对用户输入的新增需求做需求分析，技术调研（可选），编码实现，测试（单元测试+集成测试）

# rules
- **使用智能体**：MUST use sub agents FIRST！

## bugfix workflow
- **最终目标**：在目标项目内完成环境搭建、在 mini 数据集上的 1-epoch 验证与所有 bugfix，最终输出可运行的验证记录与全量训练指导文档。

### bugfix workflow stages:

#### workflow stage 1 — 环境构建与 mini 数据集准备
* **输入**：`venv.md`、`README.md`、`requirements-cpu.txt`、`requirements-gpu.txt`。
* **DO WHAT**：
    >执行 `python -m venv debug-cpu`，`pip install -r requirements-cpu.txt` 或者执行 `python -m venv debug-gpu`，`pip install -r requirements-gpu.txt`（要求用户二选一）；
    >Dataset 智能体按`venv.md`中规范抽样构建 mini数据集 并生成 `data/mini/`（样本数据、样本数、样本类别和分布等），创建notebook做可视化。
* **输出**：`env_check_report.md`（Python、包版本、可用 GPU 信息）、`data/mini/`（mini 数据集）与 `mini_dataset.md`（mini数据集使用说明和指标）。
* **WHO**：Ops 智能体 + Dataset 智能体
* **满足条件**：`env_check_report.md` 包含 `python_version`、`packages_installed`等环境安装和配置报告；mini 数据可被scripts目录下的notebook可视化。

#### workflow stage 2 — 1-epoch 验证与自动 bugfix
* **输入**：`scripts/train.py`, `data/mini/`, `DEBUG_CODE.md`。
* **DO WHAT**：
    > 根据`DEBUG_CODE.md`规范,执行`scripts/train.py`, 1-epoch 快速训练（`--fast_dev_run` 或 `--epochs 1`）；
    > Coder智能体自动采集错误信息和修复bug，生成（或更新）bugfix报告；并再次测试直至通过。
* **输出**：`bugfix_report.md`（问题清单、bug摘要、bug修复日志、测试结果）、最终训练输出checkpoint模型文件。
* **WHO**：Coder智能体。
* **满足条件**：1-epoch 运行成功，`bugfix_report.md` 中所有“关键错误”标记为已修复。

#### workflow stage 3 — 全量训练策略输出
* **输入**：`bugfix_report.md`、技术选型文档-`tech.md`。
* **DO WHAT**：根据`tech.md`生成全量训练指导文档 `full_train_guidance.md`，包括推荐的 batch\_size、learning\_rate schedule、epoch 数、checkpoint 策略、数据增强策略、分布式训练建议与监控点、各种性能配置的训练脚本示例。
* **输出**：全量训练指导文档-`full_train_guidance.md`（可直接用于 GPU 分布式训练环境）。
* **WHO**：Coder 智能体。
* **满足条件**：全量训练指导文档-`full_train_guidance.md` 包含明确的参数范围与部署步骤，且与 `tech.md` 中的资源估计一致。

---

## requirement change workflow

- **最终目标**：需求变更工作流要求对用户提出的需求修改进行分析与确认，编写需求文档，调研对应的技术与实现路径，生成任务计划，修改与扩展代码，完成测试和评审，并在目标项目中实现平稳集成与交付。

### requirement change workflow stage 1 — 需求变更分析与文档编写
* **输入**：用户自然语言变更描述（通常简短/模糊）、`requirements.md`（原始需求文档）。
* **WHO**：requirements-agent（主导），用户（回答澄清问题）。
* **DO WHAT**：
  > 解析用户提出的变更项，标注变更类型（功能增强/接口修改/性能优化/兼容性调整）；  
  > 生成/修订需求文档 `requirements_change.md`，其中包含：变更目标、影响范围、SLA 调整、约束条件变化、待澄清问题。  
* **满足条件**：需求条目与澄清问题的置信度 ≥0.6，或已人工确认。  
* **输出**：`requirements_change.md`。

### requirement change workflow stage 2 — 技术调研与影响评估
* **输入**：`requirements_change.md`、`tech.md`（原始技术文档）。  
* **WHO**：research-agent（主导），dataset-agent（当涉及 ML 算法时提供数据集支持）。  
* **DO WHAT**：
  > 基于需求变更点，评估现有架构与模块的适配性；  
  > 调研候选技术/模型/库，并分析依赖与兼容性；  
  > 若涉及 ML 算法，调用 dataset-agent 选择合适的数据集，生成 mini dataset（1-epoch 验证用）；  
  > 输出 `tech_change.md`（技术候选方案、依赖变更、兼容性风险、数据集匹配情况）。  
* **满足条件**：至少给出两个可选实现路径，说明优缺点与约束条件。  
* **输出**：`tech_change.md`。

### requirement change workflow stage 3 — 计划编排与任务拆解
* **输入**：`requirements_change.md`、`tech_change.md`、原始`task.md`与`PROJECT_BUILD_LOG.md`。  
* **WHO**：planner-agent（主导）。  
* **DO WHAT**：
  > 基于需求与技术选型，生成变更任务清单 `task_change.md`；  
  > 标注新增、修改、删除的任务，以及依赖关系与优先级；  
  > 将任务清单提交给 coder-agent 执行。  
* **满足条件**：任务清单覆盖所有变更点，且可执行。  
* **输出**：`task_change.md`。

### requirement change workflow stage 4 — 代码修改与实现
* **输入**：`task_change.md`、`tech_change.md`、`requirements_change.md`。  
* **WHO**：coder-agent（主导）。  
* **DO WHAT**：
  > 修改或扩展代码与配置文件；  
  > 更新项目文档（如 README.md）；  
  > 在`PROJECT_CHANGE_LOG.md`中记录修改点、提交记录和实现结果。  
* **满足条件**：代码能编译或运行，新增功能或修复通过基本逻辑验证。  
* **输出**：修改后的目标项目、`PROJECT_CHANGE_LOG.md`。

### requirement change workflow stage 5 — 测试与评审
* **输入**：修改后的目标项目、`task_change.md`、`PROJECT_CHANGE_LOG.md`。  
* **WHO**：tester-agent（功能与模块测试）、reviewer-agent（完整性检查与需求对齐）。  
* **DO WHAT**：
  > tester-agent 执行单元测试、集成测试与功能验证，评估回归风险；  
  > reviewer-agent 检查代码逻辑、依赖一致性、需求对齐性（对照 `requirements_change.md`）；  
  > 输出`REVIEW_CHANGE_REPORT.md`。  
* **满足条件**：所有测试用例通过，需求变更实现正确，文档与实现一致。  
* **输出**：`REVIEW_CHANGE_REPORT.md`、修订后的目标项目。

# collaborative multi-agent systems
- **requirements-agent**，需求变更（change request/requirement update）
- **research-agent**，技术选型
- **coder-agent**，编写代码
- **planner-agent**，根据需求和技术选型编排计划（plan），生成任务清单todo
- **reviewer-agent**，代码完整性检查，代码质量评审，对齐需求（`requirements.md`）和需求变更（`requirement_update_number.md`）
- **ops-agent**,创建python venv环境；创建docker环境（CPU或者GPU）
- **tester-agent**，代码测试、功能测试和模块测试 
- **dataset-agent**，找到ML算法对应的最适合的开源数据集；为1-epoch训练抽取小数据集；数据采样和可视化

# AI action rules
- **卓越工作原则**， 所有代码都要检查一遍，不可以遗漏功能，不可以偷懒，绝对不能省略代码不写，必须要完整代码
- **绝不假设目标项目已存在**，创建前必须检查
- **严格遵守'.md'中规范要求**，按规范创建目录结构，按规范编写入argparse等
- **调试优先**：GPU环境验证代码正确性后再CPU部署
- **框架双栈**：分别支持PyTorch和PaddlePaddle

# tools
- **日志规范**，打印tool use 的详细信息