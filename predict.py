"""
predict.py
----------
STAGE 6 — Inference Utility for Conformal Uncertainty Quantification

Reusable Function:
-------------------
predict_with_uncertainty(image_path) -> dict:
    - Loads model.pth and threshold.json once (cached globally).
    - Preprocesses the input image.
    - Computes softmax probabilities across all 10 classes.
    - Forms the calibrated candidate set:
          C(x) = { class_name : (1 - prob[class_name]) <= q_hat }
      An empty set remains empty; no heuristic fallback is added.
    - Returns structured dictionary:
          {
              "prediction_set": list[str],
              "confidences": dict[str, float],
              "set_size": int
          }
"""

import json
import math
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from class_mapping import ensure_same_class_mapping, validate_class_mapping
from conformal_sets import prediction_set_indices
from scripts.assemble_model import assemble_model
from paths import (
    CLASS_MAP_PATH,
    DATA_TEST,
    MODEL_PATH,
    SUPPORTED_IMAGE_EXTENSIONS,
    THRESH_PATH,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_CACHED_MODEL: nn.Module | None = None
_CACHED_THRESH: dict | None = None
_CACHED_DEVICE: torch.device | None = None


def discover_sample_images(data_root: str | Path = DATA_TEST) -> list[Path]:
    """Find supported sample images for the command-line default."""
    data_root = Path(data_root)
    return sorted(
        image
        for image in data_root.rglob("*")
        if image.is_file() and image.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _get_inference_engine() -> tuple[nn.Module, dict, torch.device]:
    """Load model and threshold metadata once and cache in memory."""
    global _CACHED_MODEL, _CACHED_THRESH, _CACHED_DEVICE

    if _CACHED_MODEL is not None and _CACHED_THRESH is not None and _CACHED_DEVICE is not None:
        return _CACHED_MODEL, _CACHED_THRESH, _CACHED_DEVICE

    if not MODEL_PATH.exists():
        # model.pth is intentionally ignored by Git; deployment builds may only
        # contain the checked-in chunks. Rebuild the exact artifact on demand.
        assemble_model()
    if not THRESH_PATH.exists():
        raise FileNotFoundError(f"Threshold file '{THRESH_PATH}' not found. Run Stage 4 first.")
    if not CLASS_MAP_PATH.exists():
        raise FileNotFoundError(f"Class mapping file '{CLASS_MAP_PATH}' not found. Run Stage 3 first.")

    with open(THRESH_PATH, "r") as f:
        thresh_data: dict = json.load(f)
    class_to_idx = validate_class_mapping(thresh_data["class_mapping"], str(THRESH_PATH))
    q_hat = thresh_data.get("q_hat")
    if (
        isinstance(q_hat, bool)
        or not isinstance(q_hat, (int, float))
        or not math.isfinite(float(q_hat))
        or not 0.0 <= float(q_hat) <= 1.0
    ):
        raise ValueError(f"Invalid q_hat in {THRESH_PATH}; expected a finite value from 0 to 1.")
    with open(CLASS_MAP_PATH, "r") as f:
        saved_class_mapping = validate_class_mapping(json.load(f), str(CLASS_MAP_PATH))
    ensure_same_class_mapping(class_to_idx, saved_class_mapping, str(CLASS_MAP_PATH))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes  = len(class_to_idx)

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(  # type: ignore[assignment]
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    _CACHED_MODEL  = model
    _CACHED_THRESH = thresh_data
    _CACHED_DEVICE = device

    return model, thresh_data, device


def predict_with_uncertainty(image_path: str | Path) -> dict:
    """
    Perform split-conformal inference for a single image.

    Args:
        image_path: Path to the image file.

    Returns:
        dict containing:
            - 'prediction_set': classes meeting the saved threshold; may be empty
            - 'prediction_set_status': whether the calibrated set is empty
            - 'confidences': dictionary of {class_name: softmax_prob} for all classes
            - 'set_size': integer size of the prediction set
            - 'leaf_screen_validated': false; the 0.20 softmax screen is heuristic only
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    model, thresh_data, device = _get_inference_engine()
    q_hat        = thresh_data["q_hat"]
    class_to_idx = thresh_data["class_mapping"]
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")
    tensor_img = inference_transform(image).unsqueeze(0).to(device)

    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        logits = model(tensor_img)
        probs  = softmax(logits).squeeze(0).cpu().numpy()

    confidences = {}
    for idx, prob in enumerate(probs):
        cls_name = idx_to_class[idx]
        confidences[cls_name] = float(prob)

    sorted_conf = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
    top_class, top_prob = sorted_conf[0]
    selected_indices = prediction_set_indices(probs, q_hat)
    prediction_set = [
        idx_to_class[index]
        for index in sorted(selected_indices, key=lambda index: float(probs[index]), reverse=True)
    ]

    # This threshold is a display heuristic, not a validated OOD detector.
    leaf_screen_passed = top_prob >= 0.20
    is_confident = (top_prob >= 0.80) and (len(prediction_set) == 1)

    return {
        "is_tomato_leaf": leaf_screen_passed,  # Compatibility alias; see leaf_screen_validated.
        "leaf_screen_passed": leaf_screen_passed,
        "leaf_screen_threshold": 0.20,
        "leaf_screen_validated": False,
        "top_class": top_class,
        "top_confidence": float(top_prob),
        "is_confident": is_confident,
        "uncertainty_level": "LOW" if is_confident else "HIGH",
        "prediction_set": prediction_set,
        "prediction_set_status": "nonempty" if prediction_set else "empty",
        "confidences": confidences,
        "set_size": len(prediction_set),
        "target_coverage": thresh_data.get("target_coverage"),
    }


import argparse

def main():
    parser = argparse.ArgumentParser(description="Conformal Prediction Single Image Inference")
    parser.add_argument("--image", type=str, help="Path to an uploaded image file")
    args = parser.parse_args()

    print("=" * 70)
    print("  STAGE 6 -- Single Image Conformal Prediction Inference")
    print("=" * 70)

    if args.image:
        image_path = Path(args.image)
        true_class = None
    else:
        sample_images = discover_sample_images()
        if not sample_images:
            print("[ERROR] No image specified with --image and no test images found in data/test.")
            return
        image_path = sample_images[0]
        true_class = image_path.parent.name

    if not image_path.exists():
        print(f"[ERROR] Specified image file not found: {image_path}")
        return

    print(f"Input Image Path : {image_path}")
    if true_class:
        print(f"True Class Label : {true_class}")
    print()

    res = predict_with_uncertainty(image_path)

    top_disease = res["top_class"].replace("Tomato_", "").replace("_", " ")
    top_pct = res["top_confidence"] * 100.0

    print("-" * 70)
    if res["is_confident"]:
        print(f"STATUS           : CONFIDENT (>= 80%)")
        print(f"PREDICTED DISEASE: {top_disease} ({top_pct:.2f}%)")
    else:
        print(f"STATUS           : NOT CONFIDENT (< 80%) -- (Model is not confident)")
        print(f"BEST CANDIDATE   : {top_disease} ({top_pct:.2f}%)")
    print("-" * 70)

    print("\nTop Class Confidences Breakdown:")
    sorted_conf = sorted(res["confidences"].items(), key=lambda x: x[1], reverse=True)
    for cls_name, prob in sorted_conf[:5]:
        clean_name = cls_name.replace("Tomato_", "").replace("_", " ")
        print(f"  * {clean_name:<32}: {prob * 100:>6.2f}%")
    print("-" * 70)

    print("\n[OK] STAGE 6 Complete: Prediction output generated successfully.\n")


if __name__ == "__main__":
    main()
