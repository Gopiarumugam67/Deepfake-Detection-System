import numpy as np
import torch

from src.data.augmentation import (
    get_train_transforms,
    get_eval_transforms,
)

img = np.full((300, 300, 3), 127, dtype=np.uint8)

print("Input:", img.shape, img.dtype)

for name, transform in [
    ("Train", get_train_transforms(224)),
    ("Eval", get_eval_transforms(224)),
]:
    output = transform(image=img)["image"]
    print(
        name,
        "| shape:", tuple(output.shape),
        "| dtype:", output.dtype,
        "| finite:", bool(torch.isfinite(output).all()),
    )
