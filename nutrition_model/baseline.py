"""Evaluate a training-mean nutrition baseline on the validation split."""
import argparse
import csv
import json
from pathlib import Path

from data_pipeline.prepare_data import TARGETS


def read_targets(path):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        expected = ["image_path", "dish_id", *TARGETS]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected columns in {path}")
        rows = [
            [float(row[target]) for target in TARGETS]
            for row in reader
        ]

    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def column_means(rows):
    return [
        sum(row[column] for row in rows) / len(rows)
        for column in range(len(TARGETS))
    ]


def mean_absolute_errors(rows, predictions):
    return [
        sum(abs(row[column] - predictions[column]) for row in rows) / len(rows)
        for column in range(len(TARGETS))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/splits/train.csv"))
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/splits/validation.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/baseline_report.json"),
    )
    args = parser.parse_args()

    training_targets = read_targets(args.train)
    validation_targets = read_targets(args.validation)

    predictions = column_means(training_targets)
    errors = mean_absolute_errors(validation_targets, predictions)

    report = {
        "baseline": "training-set mean predicted for every validation row",
        "training_rows": len(training_targets),
        "validation_rows": len(validation_targets),
        "targets": {
            target: {
                "training_mean_prediction": prediction,
                "validation_mae": error,
            }
            for target, prediction, error in zip(TARGETS, predictions, errors)
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
