# 简历描述模板

## 中文版本

**Training Log Agent｜深度学习训练日志自动分析与调参建议系统**

- 基于 FastAPI + Streamlit 构建训练日志分析 Agent，支持 `.log`、`.txt`、`.csv`、`.json` 日志上传、解析、诊断、绘图和 Markdown 报告生成。
- 设计 Rule-based Tool Agent 工作流，将日志解析、指标统计、训练问题诊断、调参建议、曲线绘制和报告生成拆分为可测试模块。
- 实现 mIoU、F1、Precision、Recall、leaf/stem IoU 等指标解析与 best/final/recent window 统计，自动识别过拟合、欠拟合、震荡、类别不平衡和 PR gap。
- 封装 mock 与 OpenAI-compatible LLM Provider，在无 API Key 环境下保持完整 Demo 可运行，同时支持后续接入 DeepSeek/Qwen/OpenAI。
- 补充 pytest、Docker Compose、一键 Demo、示例输出、架构文档和面试 Q&A，提升项目可复现性与投递展示质量。

## English Version

**Training Log Agent | Automated Deep Learning Training Log Analysis and Tuning Assistant**

- Built a FastAPI + Streamlit Agent that parses training logs, diagnoses training issues, plots metric curves, and generates Markdown reports.
- Designed a rule-based tool workflow for parsing, metric summarization, diagnostics, tuning suggestions, visualization, and report generation.
- Extracted mIoU, F1, precision, recall, leaf IoU, and stem IoU metrics to detect overfitting, underfitting, oscillation, class imbalance, and precision-recall gaps.
- Implemented mock and OpenAI-compatible LLM providers to keep demos runnable without API keys while preserving extension points for real LLMs.
- Added pytest coverage, Docker Compose, one-command demos, sample outputs, architecture docs, and interview Q&A for portfolio readiness.
