from pathlib import Path

from src.preprocessing.face_extractor import FaceExtractor, batch_extract_directory

SOURCE = Path("data/staging")
OUTPUT = Path("data/processed/faces")

def main():
    extractor = FaceExtractor(device="cpu")
    total_success = 0
    total_failed = 0

    for split in ("train", "valid", "test"):
        for label in ("real", "fake"):
            src = SOURCE / split / label
            dst = OUTPUT / split / label

            success, failed = batch_extract_directory(
                src_dir=src,
                dst_dir=dst,
                extractor=extractor,
            )

            total_success += success
            total_failed += failed
            print(
                f"{split}/{label}: "
                f"{success} faces extracted, {failed} skipped"
            )

    print(f"Total successful crops: {total_success}")
    print(f"Total skipped images: {total_failed}")
    print(f"Output directory: {OUTPUT.resolve()}")

if __name__ == "__main__":
    main()
