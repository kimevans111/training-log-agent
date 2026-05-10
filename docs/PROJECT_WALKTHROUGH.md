# 项目讲解稿

## 30 秒版本

Training Log Agent 是一个面向深度学习实验日志的 Tool Agent。它可以解析训练日志中的 loss、mIoU、F1、precision、recall 和 leaf/stem IoU，自动判断过拟合、震荡、类别不平衡等问题，并生成调参建议、训练曲线和 Markdown 报告。

## 2 分钟版本

这个项目解决的是训练日志复盘和调参建议自动化。很多深度学习实验日志很长，人工查看时容易只关注 best mIoU，而忽略后期性能下降、F1 震荡、stem IoU 低于 leaf IoU 等关键信号。

我的实现是 FastAPI + Streamlit + TrainingLogAgent。Agent 的流程是 parse、summarize、diagnose、suggest、plot、report。所有核心判断都来自确定性工具和规则，LLM 只是可选的自然语言解释层，默认 mock 模式也能完整运行。

工程上我补了 pytest、Docker Compose、一键 demo、示例数据、示例输出和面试文档，所以它不仅是脚本，而是一个可演示、可测试、可讲解的 AI 应用项目。

## 5 分钟版本

1. 背景：训练日志包含大量指标，手动复盘成本高。
2. 场景：3D plant point cloud segmentation 中 leaf/stem 类别不平衡和细结构错误很常见。
3. 架构：FastAPI 接口层，TrainingLogAgent 编排层，core tools 工具层，Streamlit 展示层。
4. 工作流：上传日志 -> 解析 metrics_series -> 统计 best/final/recent -> 诊断问题 -> 生成建议 -> 绘图 -> 报告。
5. 可解释性：每个 diagnosis 都有 evidence，面试官可以追踪到 summary 中的数值。
6. 工程质量：支持 text/CSV/JSON，带测试、Docker Compose、Demo 脚本和报告产物。
7. 后续：接 TensorBoard/W&B/MLflow，多实验对比，RAG 调参知识库，真实 LLM 解释层。

## Demo 讲解顺序

```bash
python scripts/run_demo.py
```

展示：

- 控制台 headline 和 diagnoses。
- `reports/figures/` 里的曲线图。
- `reports/` 里的 Markdown 报告。
- `docs/ARCHITECTURE.md` 里的工作流图。
