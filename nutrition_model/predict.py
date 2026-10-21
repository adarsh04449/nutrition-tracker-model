"""Predict nutrition totals for one external food image."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from data_pipeline.prepare_data import TARGETS
from nutrition_model.dataset import preprocess
from nutrition_model.model import BACKBONE, NutritionRegressor
from nutrition_model.train import choose_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("data/checkpoints/best_head.pt"))
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["targets"] != TARGETS:
        raise ValueError("Checkpoint target order does not match the dataset")
    if checkpoint["backbone"] != BACKBONE:
        raise ValueError("Checkpoint backbone does not match the model")

    model = NutritionRegressor(pretrained=False, freeze_encoder=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    with Image.open(args.image) as image:
        batch = preprocess(image).unsqueeze(0).to(device)

    target_mean = checkpoint["target_mean"].to(device)
    target_std = checkpoint["target_std"].to(device)
    with torch.inference_mode():
        normalized_prediction = model(batch)[0]
        prediction = normalized_prediction * target_std + target_mean

    result = {
        "image": str(args.image),
        "checkpoint_epoch": checkpoint["epoch"],
        "device": str(device),
        "predictions": {
            target: float(value)
            for target, value in zip(TARGETS, prediction.cpu())
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
