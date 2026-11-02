"""Rebuild the inference model from deployment-safe Git chunks."""

import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARTS_DIR = PROJECT_ROOT / "model_parts"
MODEL_PATH = PROJECT_ROOT / "model.pth"
EXPECTED_SHA256 = "062ac6ef57d65a4ea33a8bef437eca86f97ff02ac418b1e75141b4b6985f52e8"


def assemble_model() -> Path:
    """Rebuild model.pth from the checked-in chunks and verify its checksum."""
    parts = sorted(PARTS_DIR.glob("part-*.bin"))
    if not parts:
        raise FileNotFoundError(f"No model chunks found in {PARTS_DIR}")

    digest = hashlib.sha256()
    with MODEL_PATH.open("wb") as output:
        for part in parts:
            data = part.read_bytes()
            digest.update(data)
            output.write(data)

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != EXPECTED_SHA256:
        MODEL_PATH.unlink(missing_ok=True)
        raise ValueError(
            f"Model checksum mismatch: expected {EXPECTED_SHA256}, got {actual_sha256}"
        )
    print(f"Rebuilt {MODEL_PATH.name} from {len(parts)} chunks ({MODEL_PATH.stat().st_size} bytes)")
    return MODEL_PATH


def main() -> None:
    try:
        assemble_model()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
