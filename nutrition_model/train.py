"""Train the frozen-encoder nutrition regression head."""
import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.prepare_data import TARGETS
from nutrition_model.dataset import NutritionDataset
from nutrition_model.model import BACKBONE, NutritionRegressor


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def target_statistics(dataset):
    targets = torch.tensor(
        [[float(row[target]) for target in TARGETS] for row in dataset.rows],
        dtype=torch.float32,
    )
    mean = targets.mean(dim=0)
    std = targets.std(dim=0, unbiased=False)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Target statistics are not finite")
    if (std <= 0).any():
        raise ValueError("Every target must have positive variation")
    return mean, std


def train_one_epoch(model, loader, optimizer, loss_function, mean, std, device,
                    max_batches=0):
    model.train()
    loss_sum = 0.0
    example_count = 0

    for batch_number, (images, targets) in enumerate(loader, start=1):
        if max_batches and batch_number > max_batches:
            break

        images = images.to(device)
        targets = targets.to(device)
        normalized_targets = (targets - mean) / std

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = loss_function(predictions, normalized_targets)
        loss.backward()
        optimizer.step()

        batch_size = len(images)
        loss_sum += loss.item() * batch_size
        example_count += batch_size

    if example_count == 0:
        raise ValueError("Training loader produced no examples")
    return loss_sum / example_count


@torch.inference_mode()
def evaluate(model, loader, loss_function, mean, std, device, max_batches=0):
    model.eval()
    loss_sum = 0.0
    absolute_error_sum = torch.zeros(len(TARGETS), device=device)
    normalized_error_sum = torch.zeros(len(TARGETS), device=device)
    example_count = 0

    for batch_number, (images, targets) in enumerate(loader, start=1):
        if max_batches and batch_number > max_batches:
            break

        images = images.to(device)
        targets = targets.to(device)
        normalized_targets = (targets - mean) / std
        normalized_predictions = model(images)
        predictions = normalized_predictions * std + mean

        loss = loss_function(normalized_predictions, normalized_targets)
        batch_size = len(images)
        loss_sum += loss.item() * batch_size
        absolute_error_sum += (predictions - targets).abs().sum(dim=0)
        normalized_error_sum += (
            normalized_predictions - normalized_targets
        ).abs().sum(dim=0)
        example_count += batch_size

    if example_count == 0:
        raise ValueError("Validation loader produced no examples")

    return {
        "loss": loss_sum / example_count,
        "mae": (absolute_error_sum / example_count).cpu().tolist(),
        "mean_normalized_mae": (
            normalized_error_sum / example_count
        ).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-table", default="data/splits/train.csv")
    parser.add_argument("--validation-table", default="data/splits/validation.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("data/checkpoints/best_head.pt"))
    parser.add_argument("--history", type=Path,
                        default=Path("data/training_history.json"))
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="Development limit; 0 uses every training batch")
    parser.add_argument("--max-validation-batches", type=int, default=0,
                        help="Development limit; 0 uses every validation batch")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning-rate must be positive and weight-decay nonnegative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device()
    if device.type == "cpu":
        torch.set_num_threads(2)

    training_dataset = NutritionDataset(args.train_table)
    validation_dataset = NutritionDataset(args.validation_table)
    target_mean, target_std = target_statistics(training_dataset)
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = NutritionRegressor(pretrained=True, freeze_encoder=True).to(device)
    optimizer = torch.optim.AdamW(
        model.regression_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_function = nn.SmoothL1Loss()

    print(f"Device: {device}")
    print(f"Training rows: {len(training_dataset)}")
    print(f"Validation rows: {len(validation_dataset)}")
    print("Target means:", target_mean.cpu().tolist())
    print("Target standard deviations:", target_std.cpu().tolist())

    history = []
    best_score = float("inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        training_loss = train_one_epoch(
            model, training_loader, optimizer, loss_function,
            target_mean, target_std, device, args.max_train_batches,
        )
        validation = evaluate(
            model, validation_loader, loss_function,
            target_mean, target_std, device, args.max_validation_batches,
        )

        epoch_result = {
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation["loss"],
            "validation_mean_normalized_mae": validation["mean_normalized_mae"],
            "validation_mae": dict(zip(TARGETS, validation["mae"])),
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, indent=2))

        score = validation["mean_normalized_mae"]
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "target_mean": target_mean.cpu(),
                    "target_std": target_std.cpu(),
                    "targets": TARGETS,
                    "backbone": BACKBONE,
                    "epoch": epoch,
                    "validation": validation,
                },
                args.checkpoint,
            )
            print(f"Saved best checkpoint: {args.checkpoint}")

    args.history.parent.mkdir(parents=True, exist_ok=True)
    args.history.write_text(json.dumps(history, indent=2) + "\n")
    print(f"Saved training history: {args.history}")


if __name__ == "__main__":
    main()
