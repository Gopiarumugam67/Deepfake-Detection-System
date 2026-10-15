from pathlib import Path
from zipfile import ZipFile
import random
import shutil

ARCHIVE = Path.home() / "Downloads" / "archive (4).zip"
OUTPUT = Path("data") / "staging"
SEED = 42

COUNTS = {
    "train": 1750,
    "valid": 375,
    "test": 375,
}

random.seed(SEED)


def main():
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {ARCHIVE}")

    # Select image paths without extracting the full archive.
    with ZipFile(ARCHIVE) as zf:
        members = [
            name for name in zf.namelist()
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
            and name.startswith("real_vs_fake/real-vs-fake/")
        ]

        selected = {}
        for split, count in COUNTS.items():
            for label in ("real", "fake"):
                candidates = [
                    name for name in members
                    if f"/{split}/{label}/" in name
                ]
                if len(candidates) < count:
                    raise RuntimeError(
                        f"Not enough {split}/{label} images: {len(candidates)}"
                    )
                selected[(split, label)] = random.sample(candidates, count)

        total = sum(len(paths) for paths in selected.values())
        print(f"Selected {total} images from archive.")
        print("Extracting into data/staging/<split>/<real-or-fake>/")

        for (split, label), paths in selected.items():
            destination = OUTPUT / split / label
            destination.mkdir(parents=True, exist_ok=True)

            for index, member in enumerate(paths, start=1):
                filename = Path(member).name
                target = destination / filename

                with zf.open(member) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)

            print(f"{split}/{label}: {len(paths)} images")

    print("Done. Original archive and project source code were not modified.")


if __name__ == "__main__":
    main()
