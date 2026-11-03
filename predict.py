"""Model inference for the Flask application using NumPy and Pillow only."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from class_mapping import ensure_same_class_mapping, validate_class_mapping
from conformal_sets import prediction_set_indices
from numpy_resnet import NumpyResNet18
from paths import (
    CLASS_MAP_PATH,
    DATA_TEST,
    MODEL_NPZ_PATH,
    SUPPORTED_IMAGE_EXTENSIONS,
    THRESH_PATH,
)


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
_CACHED_MODEL: NumpyResNet18 | None = None
_CACHED_THRESH: dict | None = None


def discover_sample_images(data_root: str | Path = DATA_TEST) -> list[Path]:
    """Find supported sample images for the command-line default."""
    data_root = Path(data_root)
    return sorted(
        image
        for image in data_root.rglob("*")
        if image.is_file() and image.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _preprocess_image(image: Image.Image) -> np.ndarray:
    """Match torchvision's Resize(256), CenterCrop(224), and normalization."""
    width, height = image.size
    if width < height:
        resized_width = 256
        resized_height = int(256 * height / width)
    else:
        resized_height = 256
        resized_width = int(256 * width / height)

    image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    left = (resized_width - 224) // 2
    top = (resized_height - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(pixels, (2, 0, 1)).astype(np.float32, copy=False)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities)


def _get_inference_engine() -> tuple[NumpyResNet18, dict]:
    """Load the exported model and threshold metadata once per worker."""
    global _CACHED_MODEL, _CACHED_THRESH

    if _CACHED_MODEL is not None and _CACHED_THRESH is not None:
        return _CACHED_MODEL, _CACHED_THRESH

    if not MODEL_NPZ_PATH.exists():
        raise FileNotFoundError(
            f"Exported model '{MODEL_NPZ_PATH}' not found. "
            "Run scripts/export_numpy_model.py before deploying."
        )
    if not THRESH_PATH.exists():
        raise FileNotFoundError(f"Threshold file '{THRESH_PATH}' not found. Run Stage 4 first.")
    if not CLASS_MAP_PATH.exists():
        raise FileNotFoundError(f"Class mapping file '{CLASS_MAP_PATH}' not found. Run Stage 3 first.")

    with THRESH_PATH.open(encoding="utf-8") as threshold_file:
        threshold_data: dict = json.load(threshold_file)
    class_to_idx = validate_class_mapping(threshold_data["class_mapping"], str(THRESH_PATH))
    q_hat = threshold_data.get("q_hat")
    if (
        isinstance(q_hat, bool)
        or not isinstance(q_hat, (int, float))
        or not math.isfinite(float(q_hat))
        or not 0.0 <= float(q_hat) <= 1.0
    ):
        raise ValueError(f"Invalid q_hat in {THRESH_PATH}; expected a finite value from 0 to 1.")

    with CLASS_MAP_PATH.open(encoding="utf-8") as mapping_file:
        saved_class_mapping = validate_class_mapping(json.load(mapping_file), str(CLASS_MAP_PATH))
    ensure_same_class_mapping(class_to_idx, saved_class_mapping, str(CLASS_MAP_PATH))

    _CACHED_MODEL = NumpyResNet18(MODEL_NPZ_PATH)
    _CACHED_THRESH = threshold_data
    return _CACHED_MODEL, _CACHED_THRESH


def predict_with_uncertainty(image_path: str | Path) -> dict:
    """Return model scores and the calibrated conformal prediction set."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    model, threshold_data = _get_inference_engine()
    q_hat = threshold_data["q_hat"]
    class_to_idx = threshold_data["class_mapping"]
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")
    logits = model.predict(_preprocess_image(image))
    probabilities = _softmax(logits)

    confidences = {
        idx_to_class[idx]: float(probability)
        for idx, probability in enumerate(probabilities)
    }
    sorted_confidences = sorted(confidences.items(), key=lambda item: item[1], reverse=True)
    top_class, top_probability = sorted_confidences[0]
    selected_indices = prediction_set_indices(probabilities, q_hat)
    prediction_set = [
        idx_to_class[index]
        for index in sorted(selected_indices, key=lambda index: float(probabilities[index]), reverse=True)
    ]

    leaf_screen_passed = top_probability >= 0.20
    is_confident = top_probability >= 0.80 and len(prediction_set) == 1
    return {
        "is_tomato_leaf": leaf_screen_passed,
        "leaf_screen_passed": leaf_screen_passed,
        "leaf_screen_threshold": 0.20,
        "leaf_screen_validated": False,
        "top_class": top_class,
        "top_confidence": float(top_probability),
        "is_confident": is_confident,
        "uncertainty_level": "LOW" if is_confident else "HIGH",
        "prediction_set": prediction_set,
        "prediction_set_status": "nonempty" if prediction_set else "empty",
        "confidences": confidences,
        "set_size": len(prediction_set),
        "target_coverage": threshold_data.get("target_coverage"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Conformal Prediction Single Image Inference")
    parser.add_argument("--image", type=str, help="Path to an uploaded image file")
    args = parser.parse_args()

    if args.image:
        image_path = Path(args.image)
        true_class = None
    else:
        sample_images = discover_sample_images()
        if not sample_images:
            print("[ERROR] No image specified and no test images found in data/test.")
            return
        image_path = sample_images[0]
        true_class = image_path.parent.name

    if not image_path.exists():
        print(f"[ERROR] Specified image file not found: {image_path}")
        return

    result = predict_with_uncertainty(image_path)
    top_disease = result["top_class"].replace("Tomato_", "").replace("_", " ")
    top_percentage = result["top_confidence"] * 100.0
    print(f"Input Image Path : {image_path}")
    if true_class:
        print(f"True Class Label : {true_class}")
    print(f"BEST CANDIDATE   : {top_disease} ({top_percentage:.2f}%)")
    print("\nTop Class Confidences Breakdown:")
    for class_name, probability in sorted(result["confidences"].items(), key=lambda item: item[1], reverse=True)[:5]:
        clean_name = class_name.replace("Tomato_", "").replace("_", " ")
        print(f"  * {clean_name:<32}: {probability * 100:>6.2f}%")


if __name__ == "__main__":
    main()
