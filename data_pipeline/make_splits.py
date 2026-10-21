"""Create reproducible, group-aware train/validation/test CSV tables."""
import argparse
import csv
import json
import random
from pathlib import Path

from data_pipeline.prepare_data import COLUMNS


def read_ids(path):
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def timestamp(dish_id):
    return int(dish_id.removeprefix("dish_"))


def make_candidate_groups(dish_ids, max_gap):
    ordered = sorted(dish_ids, key=timestamp)
    groups = []
    current = [ordered[0]]

    for previous, current_id in zip(ordered, ordered[1:]):
        if timestamp(current_id) - timestamp(previous) <= max_gap:
            current.append(current_id)
        else:
            groups.append(current)
            current = [current_id]

    groups.append(current)
    return groups


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["image_path"] = "../" + row["image_path"]
            writer.writerow(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("data/images.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/raw/dish_ids/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-gap", type=int, default=600)
    args = parser.parse_args()

    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must be between 0 and 1")
    if args.group_gap < 0:
        parser.error("group-gap must be nonnegative")

    with args.table.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Unexpected source table columns")
        rows = list(reader)

    rows_by_id = {row["dish_id"]: row for row in rows}
    if len(rows_by_id) != len(rows):
        raise ValueError("Duplicate dish IDs in source table")

    official_train = read_ids(args.split_dir / "rgb_train_ids.txt")
    official_test = read_ids(args.split_dir / "rgb_test_ids.txt")
    if official_train & official_test:
        raise ValueError("Official train and test lists overlap")

    local_ids = set(rows_by_id)
    train_pool = local_ids & official_train
    test_ids = local_ids & official_test
    excluded_ids = local_ids - official_train - official_test

    groups = make_candidate_groups(local_ids, args.group_gap)
    train_groups = [set(group) & train_pool for group in groups]
    train_groups = [group for group in train_groups if group]

    random.Random(args.seed).shuffle(train_groups)
    validation_target = round(len(train_pool) * args.validation_fraction)
    validation_ids = set()
    for group in train_groups:
        if len(validation_ids) >= validation_target:
            break
        validation_ids.update(group)

    training_ids = train_pool - validation_ids

    split_ids = {
        "train": training_ids,
        "validation": validation_ids,
        "test": test_ids,
    }

    if split_ids["train"] & split_ids["validation"]:
        raise AssertionError("Train and validation overlap")
    if (split_ids["train"] | split_ids["validation"]) & split_ids["test"]:
        raise AssertionError("Development data overlaps official test data")
    if split_ids["train"] | split_ids["validation"] != train_pool:
        raise AssertionError("Official local training pool was not preserved")

    for group in train_groups:
        if group & training_ids and group & validation_ids:
            raise AssertionError("A candidate group crosses train and validation")

    for name, ids in split_ids.items():
        selected = [row for row in rows if row["dish_id"] in ids]
        write_table(args.output_dir / f"{name}.csv", selected)

    report = {
        "seed": args.seed,
        "validation_fraction_requested": args.validation_fraction,
        "candidate_group_gap_seconds": args.group_gap,
        "train": len(training_ids),
        "validation": len(validation_ids),
        "test": len(test_ids),
        "excluded_not_in_official_splits": len(excluded_ids),
        "candidate_groups_in_official_train_pool": len(train_groups),
        "overlap_checks": "passed",
    }
    (args.output_dir / "split_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
