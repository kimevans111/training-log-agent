"""Robust training log parser for point cloud segmentation experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*%?"
RATIO_KEYS = {
    "miou",
    "oa",
    "macc",
    "precision",
    "recall",
    "f1",
    "leaf_iou",
    "stem_iou",
    "best_miou",
    "best_f1",
}

METRIC_ALIASES: Dict[str, List[str]] = {
    "train_loss": [r"train[_\s-]*loss", r"training[_\s-]*loss", r"训练损失"],
    "val_loss": [r"val(?:idation)?[_\s-]*loss", r"valid[_\s-]*loss", r"验证损失"],
    "loss": [r"loss"],
    "miou": [r"mIoU", r"mean[_\s-]*IoU", r"avg[_\s-]*class[_\s-]*IoU", r"eval point avg class IoU"],
    "oa": [r"\bOA\b", r"overall[_\s-]*accuracy", r"eval point accuracy"],
    "macc": [r"mAcc", r"mean[_\s-]*acc(?:uracy)?", r"avg[_\s-]*class[_\s-]*acc", r"eval point avg class acc"],
    "precision": [r"mPrecision", r"precision", r"mean[_\s-]*precision"],
    "recall": [r"mRecall", r"recall", r"mean[_\s-]*recall"],
    "f1": [r"mF1", r"f1(?:[-_\s]*score)?", r"mean[_\s-]*f1"],
    "leaf_iou": [r"leaf[_\s-]*(?:class[_\s-]*)?iou", r"\bleaf\b", r"叶片", r"鍙剁墖"],
    "stem_iou": [r"stem[_\s-]*(?:class[_\s-]*)?iou", r"\bstem\b", r"茎秆", r"茎", r"鑼庣"],
}

LEAF_LABELS = ("leaf", "叶片", "叶", "鍙剁墖")
STEM_LABELS = ("stem", "茎秆", "茎", "鑼庣")

COLUMN_ALIASES: Dict[str, str] = {
    "epoch": "epoch",
    "ep": "epoch",
    "train_loss": "train_loss",
    "training_loss": "train_loss",
    "train loss": "train_loss",
    "val_loss": "val_loss",
    "valid_loss": "val_loss",
    "validation_loss": "val_loss",
    "val loss": "val_loss",
    "loss": "loss",
    "miou": "miou",
    "mean_iou": "miou",
    "mean iou": "miou",
    "mean iou (%)": "miou",
    "m_iou": "miou",
    "oa": "oa",
    "overall_accuracy": "oa",
    "overall accuracy": "oa",
    "accuracy": "oa",
    "macc": "macc",
    "mean_acc": "macc",
    "mean_accuracy": "macc",
    "precision": "precision",
    "mean_precision": "precision",
    "recall": "recall",
    "mean_recall": "recall",
    "f1": "f1",
    "f1_score": "f1",
    "mean_f1": "f1",
    "leaf_iou": "leaf_iou",
    "leaf iou": "leaf_iou",
    "stem_iou": "stem_iou",
    "stem iou": "stem_iou",
    "best_miou": "best_miou",
    "best_miou_epoch": "best_miou_epoch",
    "best_f1": "best_f1",
    "best_f1_epoch": "best_f1_epoch",
}


def parse_log_file(file_path: Path | str, normalize_percent: bool = True) -> Dict[str, Any]:
    """Parse a training log file into a unified metrics dictionary.

    The parser supports text logs, CSV files, and JSON files. Missing metrics are
    ignored rather than treated as errors. Ratio-like metrics are normalized to
    the 0-1 range when they are provided as percentages.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    suffix = path.suffix.lower()
    warnings: List[str] = []
    patterns: List[str] = []

    try:
        if suffix == ".csv":
            result = _parse_csv(path, normalize_percent)
            patterns.append("csv")
        elif suffix == ".json":
            result = _parse_json(path, normalize_percent)
            patterns.append("json")
        else:
            result = _parse_text(path, normalize_percent)
            patterns.extend(result.pop("_patterns", []))
    except Exception as exc:  # pragma: no cover - defensive guard for UI/API use
        warnings.append(f"Failed to parse log: {exc}")
        result = {"metrics_series": []}

    metrics_series = _sort_and_clean_records(result.get("metrics_series", []))
    best_miou, best_miou_epoch = _best_metric(metrics_series, "miou")
    best_f1, best_f1_epoch = _best_metric(metrics_series, "f1")

    explicit_best_miou = result.get("best_miou")
    explicit_best_f1 = result.get("best_f1")
    explicit_best_miou_epoch = result.get("best_miou_epoch")
    explicit_best_f1_epoch = result.get("best_f1_epoch")

    final_epoch = _final_epoch(metrics_series)
    parsed = {
        "file_name": path.name,
        "num_epochs": len(metrics_series),
        "final_epoch": final_epoch,
        "best_miou": explicit_best_miou if explicit_best_miou is not None else best_miou,
        "best_miou_epoch": explicit_best_miou_epoch if explicit_best_miou_epoch is not None else best_miou_epoch,
        "best_f1": explicit_best_f1 if explicit_best_f1 is not None else best_f1,
        "best_f1_epoch": explicit_best_f1_epoch if explicit_best_f1_epoch is not None else best_f1_epoch,
        "metrics_series": metrics_series,
        "warnings": warnings + result.get("warnings", []),
        "raw_patterns_matched": sorted(set(patterns + result.get("raw_patterns_matched", []))),
    }
    return parsed


def _parse_csv(path: Path, normalize_percent: bool) -> Dict[str, Any]:
    df = pd.read_csv(path)
    records: List[Dict[str, Any]] = []
    top_level: Dict[str, Any] = {"metrics_series": records, "warnings": [], "raw_patterns_matched": ["csv_columns"]}

    for _, row in df.iterrows():
        record: Dict[str, Any] = {}
        for column, value in row.items():
            key = _normalize_column_name(str(column))
            if key is None or pd.isna(value):
                continue
            parsed_value = _parse_numeric(value, key, normalize_percent)
            if parsed_value is None:
                continue
            if key in {"best_miou", "best_miou_epoch", "best_f1", "best_f1_epoch"}:
                top_level[key] = parsed_value
            else:
                record[key] = int(parsed_value) if key == "epoch" else parsed_value
        if record:
            records.append(record)

    return top_level


def _parse_json(path: Path, normalize_percent: bool) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    top_level: Dict[str, Any] = {"metrics_series": [], "warnings": [], "raw_patterns_matched": ["json_keys"]}

    if isinstance(data, list):
        candidate_records = data
    elif isinstance(data, dict):
        for key in ("best_miou", "best_miou_epoch", "best_f1", "best_f1_epoch", "final_epoch"):
            if key in data:
                metric_key = _normalize_column_name(key)
                value = _parse_numeric(data[key], metric_key or key, normalize_percent)
                top_level[key] = int(value) if key.endswith("_epoch") and value is not None else value
        candidate_records = (
            data.get("metrics_series")
            or data.get("epochs")
            or data.get("records")
            or data.get("logs")
            or []
        )
    else:
        candidate_records = []

    for item in candidate_records:
        if not isinstance(item, dict):
            continue
        record: Dict[str, Any] = {}
        for raw_key, raw_value in item.items():
            key = _normalize_column_name(str(raw_key))
            if key is None:
                continue
            value = _parse_numeric(raw_value, key, normalize_percent)
            if value is None:
                continue
            record[key] = int(value) if key == "epoch" else value
        if record:
            top_level["metrics_series"].append(record)

    return top_level


def _parse_text(path: Path, normalize_percent: bool) -> Dict[str, Any]:
    records_by_epoch: Dict[int, Dict[str, Any]] = {}
    floating_records: List[Dict[str, Any]] = []
    top_level: Dict[str, Any] = {"metrics_series": [], "warnings": [], "_patterns": []}
    current_epoch: Optional[int] = None
    current_record: Optional[Dict[str, Any]] = None
    in_class_table = False

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = _strip_log_prefix(line)

        epoch = _extract_epoch(payload)
        if epoch is not None:
            current_epoch = epoch
            current_record = records_by_epoch.setdefault(epoch, {"epoch": epoch})
            top_level["_patterns"].append("epoch_line")
            in_class_table = False

        if _is_class_table_header(payload):
            in_class_table = True
            top_level["_patterns"].append("class_table")
            continue

        if in_class_table:
            class_metrics = _extract_class_table_row(payload, normalize_percent)
            if class_metrics:
                if current_record is not None:
                    current_record.update(class_metrics)
                else:
                    record = {"source_line": line_number}
                    record.update(class_metrics)
                    floating_records.append(record)
                continue
            if "|" not in payload and not set(payload) <= {"-"}:
                in_class_table = False

        best_updates = _extract_best_metrics(payload, normalize_percent, current_epoch=current_epoch)
        if best_updates:
            top_level.update(best_updates)
            top_level["_patterns"].append("best_metric")
            if _is_metric_status_line(payload):
                continue

        if _is_metric_status_line(payload):
            continue

        metrics = _extract_metrics_from_line(payload, normalize_percent)
        if not metrics:
            continue

        if "leaf_iou" in metrics or "stem_iou" in metrics:
            top_level["_patterns"].append("class_iou")
        top_level["_patterns"].append("key_value_metrics")

        if current_record is not None:
            current_record.update(metrics)
        elif "epoch" in metrics:
            epoch_value = int(metrics.pop("epoch"))
            current_epoch = epoch_value
            current_record = records_by_epoch.setdefault(epoch_value, {"epoch": epoch_value})
            current_record.update(metrics)
        else:
            record = {"source_line": line_number}
            record.update(metrics)
            floating_records.append(record)

    top_level["metrics_series"] = list(records_by_epoch.values()) + floating_records
    return top_level


def _extract_epoch(line: str) -> Optional[int]:
    patterns = [
        r"(?:^|[\[\s|,])epoch\s*[:#]?\s*(\d+)(?:\s*/\s*\d+)?",
        r"\[epoch\s+(\d+)\]",
    ]
    for pattern in patterns:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_best_metrics(line: str, normalize_percent: bool, current_epoch: Optional[int] = None) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    for metric_name, output_key, epoch_key in (
        ("mIoU", "best_miou", "best_miou_epoch"),
        ("F1", "best_f1", "best_f1_epoch"),
    ):
        pattern = (
            rf"(?:best|最佳|新最佳)\s*{metric_name}\s*"
            rf"(?:\([^)]*\))?\s*[:=]\s*({NUMBER_RE})"
            rf"(?:\s*(?:at|@)?\s*epoch\s*[:#]?\s*(\d+))?"
        )
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            updates[output_key] = _parse_numeric(match.group(1), output_key, normalize_percent)
            if match.group(2):
                updates[epoch_key] = int(match.group(2))
            elif "新最佳" in line and current_epoch is not None:
                updates[epoch_key] = current_epoch
    return updates


def _extract_metrics_from_line(line: str, normalize_percent: bool) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for key, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            pattern = rf"(?:^|[\s,\|\[])\s*{alias}\s*(?:\([^)]*\))?\s*[:=]\s*({NUMBER_RE})"
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                value = _parse_numeric(match.group(1), key, normalize_percent)
                if value is not None:
                    metrics[key] = value
                break
    return metrics


def _strip_log_prefix(line: str) -> str:
    """Remove timestamp/logger prefix when present."""

    marker = " - INFO - "
    if marker in line:
        return line.split(marker, 1)[1].strip()
    return line


def _is_metric_status_line(line: str) -> bool:
    """Return True for checkpoint/status lines that should not update epoch metrics."""

    lowered = line.lower()
    status_tokens = ("未提升", "新最佳", "模型已保存", "训练结束", "best miou", "best f1")
    if any(token in lowered for token in status_tokens):
        return True
    return bool(re.search(r"^\s*best\s+", line, flags=re.IGNORECASE))


def _is_class_table_header(line: str) -> bool:
    return "class" in line.lower() and "iou" in line.lower() and "precision" in line.lower()


def _extract_class_table_row(line: str, normalize_percent: bool) -> Dict[str, Any]:
    if "|" not in line:
        return {}
    cells = [cell.strip() for cell in line.split("|")]
    if len(cells) < 2:
        return {}
    label = cells[0].strip()
    if not label or set(label) <= {"-"} or label.lower() == "class":
        return {}
    iou = _parse_numeric(cells[1], "miou", normalize_percent)
    if iou is None:
        return {}
    label_lower = label.lower()
    if any(token.lower() in label_lower for token in LEAF_LABELS):
        return {"leaf_iou": iou}
    if any(token.lower() in label_lower for token in STEM_LABELS):
        return {"stem_iou": iou}
    return {}


def _normalize_column_name(column: str) -> Optional[str]:
    normalized = column.strip().lower()
    normalized = normalized.replace("-", "_").replace("/", "_")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized_underscore = normalized.replace(" ", "_")

    if normalized in COLUMN_ALIASES:
        return COLUMN_ALIASES[normalized]
    if normalized_underscore in COLUMN_ALIASES:
        return COLUMN_ALIASES[normalized_underscore]

    for key, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            alias_text = _alias_to_plain(alias)
            if alias_text.lower() == normalized_underscore:
                return key
    return None


def _alias_to_plain(alias: str) -> str:
    plain = alias
    for pattern in [r"\\b", r"(?:", ")", r"\s", "[_\\s-]*", "(?:class[_\\s-]*)?"]:
        if pattern == "[_\\s-]*":
            plain = plain.replace(pattern, "_")
        elif pattern == r"\s":
            plain = plain.replace(r"\s", "_")
        else:
            plain = plain.replace(pattern, "")
    return plain.lower().strip("_")


def _parse_numeric(value: Any, key: str, normalize_percent: bool) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        raw_text = ""
    else:
        raw_text = str(value).strip()
        if raw_text == "":
            return None
        match = re.search(NUMBER_RE, raw_text)
        if not match:
            return None
        token = match.group(0).strip()
        numeric = float(token.replace("%", "").strip())

    if key in {"epoch", "best_miou_epoch", "best_f1_epoch", "final_epoch"}:
        return float(int(numeric))

    if normalize_percent and key in RATIO_KEYS:
        has_percent = "%" in raw_text
        if has_percent or (1.0 < abs(numeric) <= 100.0):
            numeric = numeric / 100.0
    return round(float(numeric), 6)


def _sort_and_clean_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for record in records:
        item = {key: value for key, value in record.items() if value is not None}
        if not item:
            continue
        if "epoch" in item:
            item["epoch"] = int(item["epoch"])
        cleaned.append(item)
    cleaned.sort(key=lambda item: (item.get("epoch") is None, item.get("epoch", item.get("source_line", 10**9))))
    return cleaned


def _best_metric(records: List[Dict[str, Any]], key: str) -> Tuple[Optional[float], Optional[int]]:
    best_value: Optional[float] = None
    best_epoch: Optional[int] = None
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_epoch = record.get("epoch")
    return best_value, best_epoch


def _final_epoch(records: List[Dict[str, Any]]) -> Optional[int]:
    epochs = [record.get("epoch") for record in records if record.get("epoch") is not None]
    return int(max(epochs)) if epochs else None
