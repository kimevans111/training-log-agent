# Training Log Agent 项目现状理解

## 项目定位

Training Log Agent 是项目二。它面向深度学习训练日志自动分析与调参建议，重点不是 RAG，而是 Rule-based Agent / Tool Agent：把日志解析、指标统计、诊断规则、绘图和报告生成组织成完整流水线。

## 已具备功能

- FastAPI 后端：上传训练日志、运行分析、基于日志回答问题、下载报告和图片。
- Streamlit 前端：上传日志、展示核心指标、诊断结果、建议、曲线和报告下载。
- Agent 编排：`TrainingLogAgent` 串联 parser、metric summary、diagnostics、suggestion engine、plotter、report generator 和 LLM provider。
- 日志解析：支持 `.log`、`.txt`、`.csv`、`.json`，能处理小数和百分比指标。
- 诊断规则：覆盖过拟合、欠拟合、训练不足、震荡、F1 不稳定、class imbalance、precision/recall gap、学习率问题。
- 报告输出：生成 Markdown 报告和多种曲线图。
- 测试：已有 parser、summary、diagnostics、report、plot 和完整 Agent flow 测试。

## 可运行部分

- `pytest` 可以跑通 12 个核心测试。
- `uvicorn app.main:app --reload` 可以启动后端。
- `streamlit run frontend/streamlit_app.py` 可以启动前端。
- `python scripts/run_demo.py` 可以直接分析样例日志。
- `docker compose up --build` 可以同时启动后端和前端。

## 仍像 Demo 的地方

- LLM provider 默认 mock；`OpenAICompatibleProvider` 已实现 HTTP 调用，但默认关闭，需要通过环境变量显式启用，保证无 API Key Demo 仍稳定。
- 诊断规则是专家规则，尚未接入学习型异常检测或历史实验数据库。
- 前端是单日志分析界面，还没有多实验管理、筛选和对比面板。
- 日志模板覆盖面有限，对 TensorBoard event、W&B、MLflow 还没有直接支持。

## 影响实习面试认可度的点

- 需要讲清楚它是 Tool Agent，不是 RAG Agent，也不是普通聊天机器人。
- 需要强调核心价值：将非结构化日志转结构化指标，再产出可执行调参建议。
- 需要展示可运行 Demo、测试、Docker Compose 和报告结果。
- 需要能解释每条诊断规则背后的深度学习训练逻辑。
- 需要说明如何启用真实 LLM provider，以及后续如何接入实验库和多实验对比。

## 最优先补强的 5 个点

1. 作品集文档和架构图，让项目定位更清楚。
2. Agent 执行轨迹，展示工具调用顺序。
3. 一键 Demo 和 Docker Compose，降低面试演示成本。
4. API 冒烟测试，补足后端可验证性。
5. 面试讲解稿、Q&A 和简历模板，帮助把项目转成实习岗位语言。
