from pathlib import Path
from src.preprocessing.face_extractor import FaceExtractor, batch_extract_directory

extractor = FaceExtractor(device="cpu")
source = Path("data/staging/train")
output = Path("data/processed/faces_smoke_test")

for label in ("real", "fake"):
    src = source / label
    dst = output / label
    dst.mkdir(parents=True, exist_ok=True)

    # Copy 10 images into a temporary input folder for this test.
    temp_input = Path("data/staging/smoke_test_input") / label
    temp_input.mkdir(parents=True, exist_ok=True)

    for image in sorted(src.glob("*.jpg"))[:10]:
        (temp_input / image.name).write_bytes(image.read_bytes())

    success, failed = batch_extract_directory(
        src_dir=temp_input,
        dst_dir=dst,
        extractor=extractor,
    )
    print(f"{label}: {success} extracted, {failed} skipped")

print(f"Smoke-test crops saved under: {output.resolve()}")
