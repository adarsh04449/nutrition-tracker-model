"""Shared deterministic preprocessing and PyTorch image/target loader."""
import csv
from pathlib import Path
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from data_pipeline.prepare_data import TARGETS


def preprocess(image, size=224):
    # Resize the whole image, preserving aspect ratio, then pad: no food is cropped.
    image = ImageOps.exif_transpose(image).convert('RGB')
    image = ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR,
                         color=(124, 116, 104), centering=(0.5, 0.5))
    return TF.normalize(TF.to_tensor(image),
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


class NutritionDataset(Dataset):
    def __init__(self, table):
        self.table = Path(table).resolve()
        with self.table.open(newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ['image_path', 'dish_id', *TARGETS]:
                raise ValueError('Unexpected columns or target order')
            self.rows = list(reader)
        if not self.rows:
            raise ValueError('Empty table')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.table.parent / row['image_path']) as image:
            x = preprocess(image)
        y = torch.tensor([float(row[k]) for k in TARGETS], dtype=torch.float32)
        if not torch.isfinite(y).all() or (y < 0).any():
            raise ValueError(f'Invalid targets for {row["dish_id"]}')
        return x, y


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--table', default='data/images.csv')
    a = p.parse_args()
    dataset = NutritionDataset(a.table)
    # Check every downloaded image, including batches that are smaller than four.
    count = 0
    for images, targets in DataLoader(dataset, batch_size=4, shuffle=False):
        assert images.shape[1:] == (3, 224, 224)
        assert targets.shape == (len(images), 4)
        assert torch.isfinite(images).all()
        count += len(images)
    print(f'PASS: {count} real images loaded with four finite targets')
    image, target = dataset[0]
    print(f'Image shape: {tuple(image.shape)}; target order: {TARGETS}')
    print(f'First targets: {target.tolist()}')
