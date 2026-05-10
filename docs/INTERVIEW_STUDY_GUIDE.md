# AI Agent / 后端 / 日志分析面试学习文档

## 1. Tool Agent 思想

Training Log Agent 的 Agent 能力来自工具编排：

- Parser 负责事实抽取。
- Summary 负责统计。
- Diagnostics 负责规则判断。
- Suggestions 负责行动建议。
- Plotter 和 Report 负责可视化与交付。

面试表达：LLM 不应该承担所有事实计算，数值类任务要尽量交给确定性工具。

## 2. 日志解析

常见日志来源：

- 训练脚本 `.log` / `.txt`。
- CSV 指标表。
- JSON metrics export。
- TensorBoard event、W&B、MLflow 是后续扩展。

解析重点：

- metric alias：同一指标可能叫 `mIoU`、`mean IoU`、`avg class IoU`。
- 百分比归一化：`92.3%` 和 `0.923` 要统一到 0-1。
- 缺失字段鲁棒性：没有 F1 时不能让系统崩溃。

## 3. 诊断规则

- 过拟合：train loss 降、val loss 升，或 best epoch 远早于 final epoch 且后期下降。
- 欠拟合：loss 高且 mIoU/F1 低。
- 训练不足：最后窗口仍明显上升。
- 震荡：recent mIoU/F1 std 过大。
- 类别不平衡：stem IoU 长期低于 leaf IoU。
- PR gap：precision 和 recall 差距过大。

## 4. 后端工程

- FastAPI endpoint 应返回结构化 JSON。
- 上传文件要限制后缀并防止路径穿越。
- 报告和图片要通过独立 endpoint 下载。
- API 冒烟测试覆盖 health、upload、analyze 和 ask。

## 5. 面试高频追问

- 为什么不用 LLM 直接读日志？因为数值解析需要确定性和可测试。
- 如何扩展新日志格式？增加 alias、模板 parser 或 adapter。
- 如何评估建议质量？用历史实验复盘、规则命中率和人工专家评估。
- 如何变成生产系统？异步任务、数据库、权限、实验 registry 和可观测性。
