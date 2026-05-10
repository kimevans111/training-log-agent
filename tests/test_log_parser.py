from pathlib import Path

from core.log_parser import parse_log_file


def test_parse_sample_log() -> None:
    path = Path("examples/sample_pointcloud_train.log")
    parsed = parse_log_file(path)

    assert parsed["num_epochs"] >= 95
    assert parsed["final_epoch"] == 100
    assert parsed["best_miou"] is not None
    assert parsed["best_f1"] is not None
    assert parsed["metrics_series"][-1]["leaf_iou"] > parsed["metrics_series"][-1]["stem_iou"]


def test_percent_values_are_normalized(tmp_path: Path) -> None:
    log_path = tmp_path / "percent.log"
    log_path.write_text(
        "Epoch 1/2 | mIoU: 92.33% | F1: 81.15% | leaf_iou: 99.91 | stem_iou: 77.50\n",
        encoding="utf-8",
    )

    parsed = parse_log_file(log_path)
    row = parsed["metrics_series"][0]

    assert row["miou"] == 0.9233
    assert row["f1"] == 0.8115
    assert row["leaf_iou"] == 0.9991
    assert row["stem_iou"] == 0.775


def test_missing_fields_do_not_crash(tmp_path: Path) -> None:
    log_path = tmp_path / "minimal.log"
    log_path.write_text("Epoch 1/3 | loss=0.4312\nEpoch 2/3 | loss=0.3901\n", encoding="utf-8")

    parsed = parse_log_file(log_path)

    assert parsed["num_epochs"] == 2
    assert parsed["metrics_series"][0]["loss"] == 0.4312
    assert parsed["best_miou"] is None


def test_parse_csv_log() -> None:
    parsed = parse_log_file(Path("examples/sample_metrics.csv"))

    assert parsed["num_epochs"] == 100
    assert parsed["metrics_series"][0]["epoch"] == 1
    assert parsed["best_miou"] is not None


def test_parse_timestamped_chinese_training_log_format(tmp_path: Path) -> None:
    log_path = tmp_path / "rapeseed_train.txt"
    log_path.write_text(
        "\n".join(
            [
                "2025-10-26 16:43:12,427 - INFO - **** Epoch 140/3000 | LR: 0.000100 ****",
                "2025-10-26 16:43:12,427 - INFO - 训练损失: 0.0315, 训练精度: 0.9904",
                "2025-10-26 16:43:12,427 - INFO - 验证 OA (总体精度): 0.9900",
                "2025-10-26 16:43:12,428 - INFO - 验证 mIoU: 0.9226 | mPrecision: 0.9692 | mRecall: 0.9483 | mF1: 0.9585",
                "2025-10-26 16:43:12,428 - INFO - Class      | IoU      | Precision  | Recall   | F1-Score   | Points",
                "2025-10-26 16:43:12,428 - INFO - -----------------------------------------------------------------------",
                "2025-10-26 16:43:12,428 - INFO - 茎秆         | 0.8558   | 0.9455     | 0.9002   | 0.9223     | 16159",
                "2025-10-26 16:43:12,428 - INFO - 叶片         | 0.9894   | 0.9930     | 0.9963   | 0.9947     | 229601",
                "2025-10-26 16:43:12,582 - INFO - 新最佳 mIoU: 0.9226. 模型已保存.",
                "2025-10-27 03:41:57,810 - INFO - **** Epoch 240/3000 | LR: 0.000099 ****",
                "2025-10-27 03:41:57,810 - INFO - 训练损失: 0.0279, 训练精度: 0.9916",
                "2025-10-27 03:41:57,810 - INFO - 验证 mIoU: 0.8614 | mPrecision: 0.9497 | mRecall: 0.8956 | mF1: 0.9207",
                "2025-10-27 03:41:57,811 - INFO - mIoU 未提升. 最佳 mIoU: 0.9226. 耐心: 100/100",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_log_file(log_path)
    final_row = parsed["metrics_series"][-1]

    assert parsed["best_miou"] == 0.9226
    assert parsed["best_miou_epoch"] == 140
    assert final_row["miou"] == 0.8614
    assert final_row["f1"] == 0.9207
    assert parsed["metrics_series"][0]["stem_iou"] == 0.8558
    assert parsed["metrics_series"][0]["leaf_iou"] == 0.9894
