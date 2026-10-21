"""Inspect locally available dishes absent from official RGB split lists."""
import csv
from pathlib import Path


TABLE = Path("data/images.csv")
SPLIT_DIR = Path("data/raw/dish_ids/splits")
METADATA_DIR = Path("data/raw/metadata")


def read_ids(path):
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main():
    train = read_ids(SPLIT_DIR / "rgb_train_ids.txt")
    test = read_ids(SPLIT_DIR / "rgb_test_ids.txt")

    with TABLE.open(newline="") as file:
        rows = list(csv.DictReader(file))

    local = {row["dish_id"] for row in rows}
    missing = sorted(local - train - test)
    image_paths = {row["dish_id"]: Path("data") / row["image_path"] for row in rows}

    metadata_ids = set()
    for path in sorted(METADATA_DIR.glob("dish_metadata_cafe*.csv")):
        with path.open(newline="") as file:
            metadata_ids.update(row[0] for row in csv.reader(file) if row)

    missing_images = [dish_id for dish_id in missing if not image_paths[dish_id].is_file()]
    missing_metadata = [dish_id for dish_id in missing if dish_id not in metadata_ids]

    print(f"Missing from official lists: {len(missing)}")
    print(f"Missing image files: {len(missing_images)}")
    print(f"Missing metadata rows: {len(missing_metadata)}")
    print()
    print("First 20 missing IDs:")
    for dish_id in missing[:20]:
        print(dish_id)

    if not missing_images and not missing_metadata:
        print()
        print("All missing-list dishes have local images and metadata.")


if __name__ == "__main__":
    main()
