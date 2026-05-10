# Training Log Agent 作品集说明

## 这个项目解决什么问题

深度学习训练日志通常很长，包含 loss、mIoU、F1、precision、recall、class-wise IoU 等指标。人工分析时容易只看 best metric，忽略过拟合、训练震荡、类别不平衡和 precision/recall 矛盾。

Training Log Agent 将训练日志自动解析为结构化指标，并通过规则诊断和调参建议生成完整 Markdown 报告，适合用于科研实验复盘和模型调参辅助。

## 为什么不是普通聊天机器人

本项目不依赖 LLM 直接“猜”结论，而是先运行确定性工具：

- Parser 从日志中抽取 metric series。
- Summary 计算 best/final/recent window 统计。
- Diagnostics 根据训练动态触发规则。
- Suggestion engine 把问题映射到调参建议和下一组实验。
- Plotter 生成曲线图。
- Report generator 汇总成 Markdown。

LLM provider 只是可选问答增强，默认 mock 模式保证项目无 API Key 可运行。

## 和 AI Agent 实习岗位的关系

- Agent 应用开发：把自然语言问题和文件输入转成多工具流水线。
- 后端工程：FastAPI、文件上传、路径安全、报告下载。
- AI 应用工程：训练日志结构化、诊断规则、自动报告。
- 大模型应用：OpenAI-compatible provider 可插拔，mock fallback 保证 Demo 稳定。
- 工程质量：pytest、Docker Compose、示例数据、示例输出和文档。

## 技术栈说明

| 层级 | 技术 |
| --- | --- |
| Backend | FastAPI, Pydantic, Uvicorn |
| Agent | Rule-based tool orchestration |
| Parser | regex, pandas, JSON loader |
| Metrics | numpy statistics |
| Visualization | Matplotlib |
| Frontend | Streamlit |
| Report | Markdown |
| Test/Deploy | pytest, Docker, Docker Compose |

## 核心功能截图占位说明

建议补充截图到 `docs/assets/`：

- `01_upload_log.png`：上传日志文件。
- `02_metric_summary.png`：best/final metrics 面板。
- `03_diagnoses.png`：诊断问题和建议。
- `04_generated_curves.png`：loss、mIoU、F1、class IoU、PR 曲线。
- `05_markdown_report.png`：报告内容。

## Demo 流程

1. 运行 `docker compose up --build`。
2. 打开 Streamlit：`http://localhost:8501`。
3. 上传 `examples/sample_pointcloud_train.log`。
4. 输入问题：`What should I tune next if stem IoU is lower than leaf IoU?`
5. 点击 Analyze Log。
6. 展示指标摘要、诊断、调参建议、下一步实验、曲线图和报告下载。

命令行 Demo：

```bash
python scripts/run_demo.py
```

## 面试时如何介绍

一句话：

> 这是一个面向深度学习训练日志的 Tool Agent，能够把非结构化训练日志解析成结构化指标，并自动完成诊断、调参建议、绘图和报告生成。

讲解顺序：

- 背景：实验日志长、指标多、人工复盘低效。
- 架构：FastAPI + TrainingLogAgent + core tools + Streamlit。
- 工作流：parse -> summarize -> diagnose -> suggest -> plot -> report。
- 结果：输出 structured JSON、PNG curves、Markdown report。
- 不足：当前主要是规则系统；真实 LLM provider 已封装但默认关闭，后续重点是历史实验库和多实验对比。

## 项目亮点

- 规则可解释：每条诊断都有 evidence 和 suggestion。
- 无 API Key 可运行：mock LLM fallback。
- 支持多格式日志：文本、CSV、JSON。
- 面向具体科研场景：3D plant point cloud segmentation。
- 工程闭环完整：API、UI、测试、Docker、文档、示例输出。

## 项目不足与后续优化

- 支持 TensorBoard event、W&B export、MLflow metrics。
- 增加多实验对比、自动 ablation table 和实验 registry。
- 增加 HTML/PDF 报告导出。
- 引入真实 LLM 对诊断结果做自然语言解释，但保留结构化规则作为事实来源。
- 将规则阈值配置化，支持不同任务/数据集定制。
