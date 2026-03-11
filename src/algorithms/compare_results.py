"""Compare model results from report files and write a precision/accuracy comparison report."""

from __future__ import annotations

import re
import sys

from algorithms.config import REPORT_DIR

# Model names must match run scripts and report filenames.
MODELS: tuple[str, ...] = ("logreg", "nn", "xgboost")

# sklearn classification_report "weighted avg" line: precision is the first number.
WEIGHTED_AVG_PRECISION_RE = re.compile(
    r"weighted avg\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+\d+",
    re.IGNORECASE,
)

# Top-level "Accuracy: 0.7943" style line near the start of each report.
ACCURACY_RE = re.compile(r"Accuracy:\s*([\d.]+)", re.IGNORECASE)


def parse_weighted_precision(report_text: str) -> float | None:
    """Extract weighted average precision from a classification report string."""
    for line in report_text.splitlines():
        m = WEIGHTED_AVG_PRECISION_RE.search(line)
        if m:
            return float(m.group(1))
    return None


def parse_accuracy(report_text: str) -> float | None:
    """Extract overall accuracy from the explicit 'Accuracy:' line in the report."""
    for line in report_text.splitlines():
        m = ACCURACY_RE.search(line)
        if m:
            return float(m.group(1))
    return None


def collect_metrics() -> dict[str, tuple[float, float]]:
    """
    Read report files and return (precision, accuracy) per model.

    Returns:
        metrics[model] = (weighted_precision, accuracy)
    """
    metrics: dict[str, tuple[float, float]] = {}
    for model in MODELS:
        report_file = REPORT_DIR / f"{model}_report.txt"
        if not report_file.is_file():
            print(
                f"Warning: report file not found: {report_file}",
                file=sys.stderr,
            )
            continue
        text = report_file.read_text()
        precision = parse_weighted_precision(text)
        accuracy = parse_accuracy(text)
        if precision is None:
            print(
                f"Warning: could not parse precision from {report_file}",
                file=sys.stderr,
            )
        if accuracy is None:
            print(
                f"Warning: could not parse accuracy from {report_file}",
                file=sys.stderr,
            )
        if precision is not None and accuracy is not None:
            metrics[model] = (precision, accuracy)
    return metrics


def main() -> None:
    """Compare results and write precision/accuracy report to the report directory."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = collect_metrics()

    if not metrics:
        print("No metrics parsed; aborting comparison.", file=sys.stderr)
        return

    # Order models from highest to lowest precision and accuracy.
    ordered_by_precision = sorted(metrics.items(), key=lambda x: x[1][0], reverse=True)
    ordered_by_accuracy = sorted(metrics.items(), key=lambda x: x[1][1], reverse=True)

    out_path = REPORT_DIR / "model_comparison.txt"
    lines = [
        "Precision and accuracy of all three models",
        "(precision = weighted avg from classification report; accuracy = overall Accuracy line)",
        "",
    ]
    lines.append("Model         Precision  Accuracy")
    lines.append("-" * 34)
    for model, (precision, accuracy) in ordered_by_precision:
        lines.append(f"{model:<13} {precision:.4f}   {accuracy:.4f}")
    lines.append("")
    lines.append("Ranking by precision (highest to lowest):")
    for rank, (model, (precision, _)) in enumerate(ordered_by_precision, start=1):
        lines.append(f"  {rank}. {model}: {precision:.4f}")
    lines.append("")
    lines.append("Ranking by accuracy (highest to lowest):")
    for rank, (model, (_, accuracy)) in enumerate(ordered_by_accuracy, start=1):
        lines.append(f"  {rank}. {model}: {accuracy:.4f}")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Precision/accuracy comparison written to {out_path}")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
