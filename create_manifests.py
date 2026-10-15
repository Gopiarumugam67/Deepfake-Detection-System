from pathlib import Path
import pandas as pd

SOURCE = Path("data/processed/faces")
OUTPUT = Path("data/splits")
SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}
CLASSES = ("real", "fake")
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for source_split, manifest_name in SPLITS.items():
        records = []

        for label in CLASSES:
            folder = SOURCE / source_split / label
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing directory: {folder}")

            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    records.append({
                        "filepath": str(path.resolve()),
                        "label": label,
                    })

        if not records:
            raise RuntimeError(f"No images found for {source_split}")

        df = pd.DataFrame(records)
        manifest = OUTPUT / f"{manifest_name}.csv"
        df.to_csv(manifest, index=False)

        print(f"{manifest_name}: {len(df)} images")
        print(df["label"].value_counts().to_dict())
        print(f"Saved: {manifest.resolve()}")


if __name__ == "__main__":
    main()
