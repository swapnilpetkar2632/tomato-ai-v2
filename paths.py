"""Project-relative paths and shared image-file configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_TEST = DATA_DIR / "test"
MODEL_PATH = PROJECT_ROOT / "model.pth"
MODEL_NPZ_PATH = PROJECT_ROOT / "model.npz"
THRESH_PATH = PROJECT_ROOT / "threshold.json"
CLASS_MAP_PATH = PROJECT_ROOT / "class_mapping.json"
RESULTS_PATH = PROJECT_ROOT / "results.txt"
OOD_RESULTS_PATH = PROJECT_ROOT / "ood_results.txt"

SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
)
