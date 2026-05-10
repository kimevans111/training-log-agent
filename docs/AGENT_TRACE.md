# Agent 执行轨迹示例

用户输入：

```text
log_file_path = "examples/sample_pointcloud_train.log"
question = "What should I tune next if stem IoU is lower than leaf IoU?"
```

执行轨迹：

| Step | Action | Tool | Output |
| --- | --- | --- | --- |
| 1 | 读取日志 | `parse_log_file` | `metrics_series`, best metrics, warnings |
| 2 | 统计指标 | `generate_metric_summary` | best/final metrics, trend, class gap |
| 3 | 诊断问题 | `diagnose_training` | overfitting、class imbalance、oscillation 等 |
| 4 | 生成建议 | `generate_suggestions` | priority suggestions 和 next experiments |
| 5 | 绘图 | `generate_plots` | loss、mIoU、F1、class IoU、PR 曲线 |
| 6 | 生成报告 | `generate_report` | Markdown report |
| 7 | 回答问题 | `MockLLMProvider` 或真实 provider | 基于 summary/diagnoses/suggestions 的回答 |
| 8 | 返回结果 | API response | summary、diagnoses、suggestions、figures、report_path、answer |

结构化返回重点：

```json
{
  "summary": {"headline": "Parsed 100 epochs..."},
  "diagnoses": [{"type": "class_imbalance", "severity": "medium"}],
  "suggestions": {"priority_suggestions": ["Use class-weighted CE/Dice/Focal loss..."]},
  "figures": ["reports/figures/loss_curve.png"],
  "report_path": "reports/training_log_report_YYYYMMDD_HHMMSS.md"
}
```

工程说明：每一步都有明确输入输出，诊断结论可追溯到具体指标，不依赖 LLM 幻觉。
