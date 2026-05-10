# Training Log Agent

Training Log Agent 是一个面向深度学习实验的训练日志自动分析与调参建议系统。它可以读取 `.log`、`.txt`、`.csv`、`.json` 训练日志，解析 epoch、loss、mIoU、F1、Precision、Recall、leaf/stem IoU 等指标，自动绘制训练曲线，诊断过拟合、欠拟合、震荡、不收敛、类别不平衡等问题，并生成 Markdown 实验分析报告。

这个项目适合用于展示 AI Agent 工具调用、深度学习实验自动分析、训练日志解析、自动报告生成和科研自动化应用场景。

## 作品集文档

本项目已经补充面向实习投递和面试讲解的系统文档：

- `docs/PROJECT_REVIEW.md`：当前项目理解、可运行能力、Demo 感和优先补强点。
- `docs/README_PORTFOLIO.md`：作品集级项目介绍。
- `docs/ARCHITECTURE.md`：架构、数据流、Agent 工作流、Tool Calling 和为什么本项目不是 RAG 主线。
- `docs/AGENT_TRACE.md`：一次日志分析的 Agent 执行轨迹。
- `docs/PROJECT_WALKTHROUGH.md`：30 秒、2 分钟、5 分钟讲解稿。
- `docs/INTERVIEW_STUDY_GUIDE.md`：AI Agent / 后端 / 日志分析面试学习文档。
- `docs/INTERVIEW_QA.md`：常见面试问答。
- `docs/RESUME_TEMPLATE.md`：中文和英文简历描述模板。
- `docs/DEMO_OUTPUT.md`：命令行 Demo 示例输出。

## 项目亮点

- 面向 3D plant point cloud semantic segmentation 场景设计，关注 leaf/stem segmentation、instance clustering 和 phenotyping 前处理。
- 规则 Agent 可在无 API Key 环境完整运行，同时保留 OpenAI-compatible、DeepSeek、Qwen 等 LLM Provider 扩展接口。
- 支持多格式日志解析，自动处理百分数和小数指标。
- 自动生成 loss、mIoU、F1、class-wise IoU、Precision/Recall 曲线。
- 输出可下载 Markdown 报告，包含 detected issues、next experiments 和 generated figures。

## 功能列表

- 日志上传与保存到 `uploads/`
- `.log` / `.txt` key-value 日志解析
- CSV 指标表解析
- JSON `metrics_series` / `epochs` / `records` 解析
- 指标摘要：best mIoU、best F1、final mIoU、final F1、最后 10% epoch 均值和波动
- 诊断规则：过拟合、欠拟合、训练不足、F1 震荡、类别不平衡、学习率过大/过小、Precision/Recall 矛盾
- 调参建议：学习率、warmup、batch size、gradient clipping、EMA、class-balanced sampling、focal loss、边界损失等
- FastAPI 后端和 Streamlit 前端
- pytest 测试覆盖解析、摘要、诊断、报告、绘图和完整 Agent 流程

## 目录结构

```text
Training-Log-Agent/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── agent/
│   ├── __init__.py
│   └── training_log_agent.py
├── core/
│   ├── __init__.py
│   ├── log_parser.py
│   ├── metric_summary.py
│   ├── diagnostics.py
│   ├── suggestion_engine.py
│   ├── plotter.py
│   └── report_generator.py
├── llm/
│   ├── __init__.py
│   └── provider.py
├── frontend/
│   └── streamlit_app.py
├── examples/
│   ├── sample_pointcloud_train.log
│   └── sample_metrics.csv
├── uploads/
│   └── .gitkeep
├── reports/
│   ├── .gitkeep
│   └── figures/
│       └── .gitkeep
├── tests/
├── docker/
│   └── Dockerfile
├── docs/
├── scripts/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── requirements.txt
├── README.md
└── AGENTS.md
```

## 安装依赖

建议使用 Python 3.10+。

```bash
cd Training-Log-Agent
pip install -r requirements.txt
```

## 启动 FastAPI 后端

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API 文档启动后访问：

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

## 启动 Streamlit 前端

```bash
streamlit run frontend/streamlit_app.py
```

打开页面后上传 `examples/sample_pointcloud_train.log`，点击 Analyze Log，即可看到指标摘要、诊断结果、调参建议、训练曲线和报告下载按钮。

## 一键 Demo

无需启动 Web UI，直接运行样例日志分析：

```bash
python scripts/run_demo.py
```

Windows PowerShell helper：

```powershell
.\scripts\run_backend.ps1
.\scripts\run_frontend.ps1
```

## 使用示例

```python
from pathlib import Path
from agent.training_log_agent import TrainingLogAgent

agent = TrainingLogAgent()
result = agent.analyze(
    Path("examples/sample_pointcloud_train.log"),
    user_question="Why is stem IoU lower than leaf IoU?",
)

print(result["summary"]["headline"])
print(result["report_path"])
```

## API 文档说明

- `GET /`: 服务信息
- `GET /health`: 健康检查
- `POST /upload`: 上传 `.log`、`.txt`、`.csv`、`.json`
- `POST /analyze-log`: 根据本地日志路径运行完整分析
- `POST /ask-about-log`: 分析日志并回答用户问题
- `GET /reports/{filename}`: 下载 Markdown 报告
- `GET /figures/{filename}`: 下载生成的曲线图

`POST /analyze-log` 示例请求：

```json
{
  "log_file_path": "examples/sample_pointcloud_train.log",
  "user_question": "What should I tune next?",
  "config": {
    "normalize_percent": true
  }
}
```

## 示例日志格式

格式 A：

```text
Epoch 1/3000 | train_loss: 1.2345 | val_loss: 1.0021 | mIoU: 0.4567 | F1: 0.5012 | OA: 0.8123 | leaf_iou: 0.6501 | stem_iou: 0.2634
```

格式 B：

```text
[Epoch 25] loss=0.4312, miou=0.8123, precision=0.9012, recall=0.8765, f1=0.8887
```

格式 C：

```text
eval point avg class IoU: 0.9233
eval point accuracy: 0.9812
eval point avg class acc: 0.9412
Best mIoU: 0.9233 at epoch 1498
Best F1: 0.8115 at epoch 1724
```

格式 D：

```text
Class IoU:
Leaf: 99.91
Stem: 99.57
```

格式 E：

```text
Epoch: 100, Train Loss: 0.256, Val Loss: 0.301, Mean IoU: 0.8742, Mean Precision: 0.9123, Mean Recall: 0.8844, Mean F1: 0.8981
```

## 如何运行测试

```bash
pytest
```

测试覆盖：

- 日志解析
- 百分数转小数
- 缺失字段鲁棒性
- 指标摘要
- 过拟合、震荡、类别不平衡诊断
- Markdown 报告生成
- 完整 Agent 流程和绘图保存
- FastAPI health/upload/analyze/ask 接口冒烟测试

## Docker Compose

同时启动 FastAPI 后端和 Streamlit 前端：

```bash
docker compose up --build
```

打开：

- API: <http://localhost:8000/docs>
- Streamlit: <http://localhost:8501>

## 当前诊断规则

- 过拟合：train loss 下降但 val loss 上升，或 best epoch 远早于 final epoch 且后期指标下降。
- 欠拟合：train/val loss 都偏高，且 mIoU/F1 长期偏低。
- 训练不足：最后 10% epoch 指标仍明显上升。
- 震荡：最近窗口 mIoU 或 F1 标准差过大。
- F1 不稳定：F1 波动明显大于 mIoU。
- 类别不平衡：stem IoU 长期低于 leaf IoU，默认 gap > 0.15。
- Precision/Recall 矛盾：高 Precision 低 Recall 代表漏检，高 Recall 低 Precision 代表误检。
- 学习率推断：震荡可能 lr 过大，平台期或缓慢提升可能 lr 过小或 scheduler 不合适。

## 后续 TODO

- 接入真实 OpenAI-compatible / DeepSeek / Qwen Chat API。
- 支持 TensorBoard event file、Weights & Biases export、MLflow metrics。
- 增加 class-wise confusion matrix 和 per-class F1 解析。
- 增加多实验对比报告和自动 ablation table。
- 增加 PDF/HTML 报告导出。
- 提供 Docker Compose，同时启动 FastAPI 和 Streamlit。
- 为 Plant-GeoAT、PointNet++、DGCNN、PointTransformerV3、PointNeXt 等模型增加日志模板。
