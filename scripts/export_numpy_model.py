"""Export the trained PyTorch checkpoint for the lightweight web runtime."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from paths import MODEL_NPZ_PATH, MODEL_PATH


def export_model(source: str | Path = MODEL_PATH, destination: str | Path = MODEL_NPZ_PATH) -> Path:
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        raise FileNotFoundError(f"Trained checkpoint not found: {source}")

    state_dict = torch.load(source, map_location="cpu", weights_only=True)
    arrays = {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in state_dict.items()
        if tensor.ndim > 0 and "num_batches_tracked" not in name
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    print(f"Exported {len(arrays)} tensors to {destination} ({destination.stat().st_size} bytes)")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model.pth for NumPy inference")
    parser.add_argument("--source", type=Path, default=MODEL_PATH)
    parser.add_argument("--destination", type=Path, default=MODEL_NPZ_PATH)
    args = parser.parse_args()
    export_model(args.source, args.destination)


if __name__ == "__main__":
    main()
