"""Check candidate timestamp sessions against official RGB train/test IDs."""
import csv
from pathlib import Path


TABLE = Path("data/images.csv")
SPLIT_DIR = Path("data/raw/dish_ids/splits")
SESSION_GAP_SECONDS = 600


def read_ids(path):
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def local_dish_ids():
    with TABLE.open(newline="") as file:
        return sorted(
            {row["dish_id"] for row in csv.DictReader(file)},
            key=lambda dish_id: int(dish_id.removeprefix("dish_")),
        )


def candidate_sessions(dish_ids):
    sessions = []
    current = [dish_ids[0]]
    for previous, dish_id in zip(dish_ids, dish_ids[1:]):
        gap = int(dish_id.removeprefix("dish_")) - int(previous.removeprefix("dish_"))
        if gap <= SESSION_GAP_SECONDS:
            current.append(dish_id)
        else:
            sessions.append(current)
            current = [dish_id]
    sessions.append(current)
    return sessions


def main():
    train = read_ids(SPLIT_DIR / "rgb_train_ids.txt")
    test = read_ids(SPLIT_DIR / "rgb_test_ids.txt")
    local = set(local_dish_ids())
    sessions = candidate_sessions(sorted(local, key=lambda dish_id: int(dish_id.removeprefix("dish_"))))

    mixed = []
    local_train = local & train
    local_test = local & test

    for session in sessions:
        labels = {"train" if dish_id in train else "test" if dish_id in test else "unknown"
                  for dish_id in session}
        if len(labels) > 1:
            mixed.append((session, labels))

    print(f"Official train IDs: {len(train)}")
    print(f"Official test IDs: {len(test)}")
    print(f"Local RGB dishes: {len(local)}")
    print(f"Local train dishes: {len(local_train)}")
    print(f"Local test dishes: {len(local_test)}")
    print(f"Local IDs missing from official lists: {len(local - train - test)}")
    print(f"Candidate sessions crossing official boundaries: {len(mixed)}")

    if mixed:
        print("First mixed session:", mixed[0])
    else:
        print("PASS: every local candidate session stays within one official split")


if __name__ == "__main__":
    main()
