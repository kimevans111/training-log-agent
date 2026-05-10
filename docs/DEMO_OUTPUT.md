# Demo Output Snapshot

This is the expected shape of `python scripts/run_demo.py`.

```text
== Training Log Agent Demo ==
Parsed 100 epochs up to epoch 100. Best mIoU=0.8866 at epoch 86, best F1=0.8840 at epoch 78.

-- Diagnoses --
- overfitting (medium): train_loss_delta=..., final_mIoU_drop=...
- class_imbalance (medium): Mean leaf IoU=..., stem IoU=..., gap=...
- low_recall (medium): Precision=..., recall=...

-- Priority Suggestions --
- Enable early stopping around the best mIoU/F1 epoch and keep the checkpoint with the strongest validation metric.
- Use class-weighted CE/Dice/Focal loss with a higher stem weight.

Figures: 5
Report: reports/training_log_report_YYYYMMDD_HHMMSS.md
```

Exact values may vary if the sample log changes.
