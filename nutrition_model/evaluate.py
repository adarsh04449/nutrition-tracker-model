"""Evaluate the best checkpoint on validation or the official test split."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.prepare_data import TARGETS
from nutrition_model.dataset import NutritionDataset
from nutrition_model.model import BACKBONE, NutritionRegressor
from nutrition_model.train import choose_device, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("data/checkpoints/best_head.pt"))
    parser.add_argument("--split", choices=("validation", "test"),
                        default="validation",
                        help="Split to evaluate; test should be used only for a final benchmark.")
    parser.add_argument("--baseline", type=Path,
                        default=Path("data/baseline_report.json"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    table = Path(f"data/splits/{args.split}.csv")
    output = args.output or Path(f"data/{args.split}_report.json")

    device = choose_device()
    if device.type == "cpu":
        torch.set_num_threads(2)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint["targets"] != TARGETS:
        raise ValueError("Checkpoint target order does not match the dataset")
    if checkpoint["backbone"] != BACKBONE:
        raise ValueError("Checkpoint backbone does not match the model")

    model = NutritionRegressor(pretrained=False, freeze_encoder=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    dataset = NutritionDataset(table)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    target_mean = checkpoint["target_mean"].to(device)
    target_std = checkpoint["target_std"].to(device)
    metrics = evaluate(
        model,
        loader,
        nn.SmoothL1Loss(),
        target_mean,
        target_std,
        device,
    )

    baseline = json.loads(args.baseline.read_text())
    baseline_means = torch.tensor(
        [baseline["targets"][target]["training_mean_prediction"] for target in TARGETS],
        dtype=torch.float32,
    )
    # Calculate the mean-prediction baseline on whichever split was selected.
    targets = torch.stack([dataset[index][1] for index in range(len(dataset))])
    baseline_maes = torch.mean(torch.abs(targets - baseline_means), dim=0).tolist()
    comparisons = {}
    for target, model_mae, baseline_mae in zip(TARGETS, metrics["mae"], baseline_maes):
        improvement = baseline_mae - model_mae
        comparisons[target] = {
            "baseline_mae": baseline_mae,
            "model_mae": model_mae,
            "mae_reduction": improvement,
            "percent_reduction": 100 * improvement / baseline_mae,
        }

    report = {
        "split": args.split,
        "rows": len(dataset),
        "checkpoint_epoch": checkpoint["epoch"],
        "device": str(device),
        "loss": metrics["loss"],
        "mean_normalized_mae": metrics["mean_normalized_mae"],
        "comparison_to_training_mean_baseline": comparisons,
        "official_test_used": args.split == "test",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
