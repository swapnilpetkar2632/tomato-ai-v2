"""Small NumPy inference implementation for the exported ResNet18 model.

The training pipeline still uses PyTorch, but the web application loads this
portable weight file instead of importing torch or torchvision. Keeping the
runtime implementation here makes the deployment requirements small and
CPU-only while preserving the trained model architecture.
"""

from pathlib import Path

import numpy as np


class NumpyResNet18:
    """Run the project's ResNet18 classifier with NumPy only."""

    def __init__(self, weights_path: str | Path):
        self._archive = np.load(Path(weights_path), allow_pickle=False)
        self._weights = {
            name: self._archive[name].astype(np.float32, copy=False)
            for name in self._archive.files
        }

    def _weight(self, name: str) -> np.ndarray:
        try:
            return self._weights[name]
        except KeyError as error:
            raise ValueError(f"Missing exported model weight: {name}") from error

    @staticmethod
    def _conv2d(
        image: np.ndarray,
        weight: np.ndarray,
        stride: int = 1,
        padding: int = 0,
    ) -> np.ndarray:
        if padding:
            image = np.pad(
                image,
                ((0, 0), (padding, padding), (padding, padding)),
                mode="constant",
            )

        kernel_height, kernel_width = weight.shape[2:]
        windows = np.lib.stride_tricks.sliding_window_view(
            image,
            (kernel_height, kernel_width),
            axis=(1, 2),
        )
        windows = windows[:, ::stride, ::stride, :, :]
        return np.einsum(
            "ihwkl,oikl->ohw",
            windows,
            weight,
            optimize=True,
        ).astype(np.float32, copy=False)

    @staticmethod
    def _batch_norm(
        image: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        running_mean: np.ndarray,
        running_var: np.ndarray,
    ) -> np.ndarray:
        scale = weight[:, None, None]
        offset = bias[:, None, None]
        mean = running_mean[:, None, None]
        variance = running_var[:, None, None]
        return ((image - mean) / np.sqrt(variance + 1e-5) * scale + offset).astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _relu(image: np.ndarray) -> np.ndarray:
        return np.maximum(image, 0.0).astype(np.float32, copy=False)

    @staticmethod
    def _max_pool(image: np.ndarray) -> np.ndarray:
        padded = np.pad(image, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=-np.inf)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
        return windows[:, ::2, ::2, :, :].max(axis=(-1, -2)).astype(np.float32, copy=False)

    def _normalise_batch_norm(self, image: np.ndarray, prefix: str) -> np.ndarray:
        return self._batch_norm(
            image,
            self._weight(f"{prefix}.weight"),
            self._weight(f"{prefix}.bias"),
            self._weight(f"{prefix}.running_mean"),
            self._weight(f"{prefix}.running_var"),
        )

    def _block(self, image: np.ndarray, prefix: str, stride: int = 1) -> np.ndarray:
        identity = image
        output = self._conv2d(
            image,
            self._weight(f"{prefix}.conv1.weight"),
            stride=stride,
            padding=1,
        )
        output = self._normalise_batch_norm(output, f"{prefix}.bn1")
        output = self._relu(output)
        output = self._conv2d(output, self._weight(f"{prefix}.conv2.weight"), padding=1)
        output = self._normalise_batch_norm(output, f"{prefix}.bn2")

        if stride != 1 or identity.shape[0] != output.shape[0]:
            identity = self._conv2d(
                identity,
                self._weight(f"{prefix}.downsample.0.weight"),
                stride=stride,
            )
            identity = self._normalise_batch_norm(identity, f"{prefix}.downsample.1")

        return self._relu(output + identity)

    def _layer(self, image: np.ndarray, layer_number: int, stride: int) -> np.ndarray:
        prefix = f"layer{layer_number}"
        image = self._block(image, f"{prefix}.0", stride=stride)
        return self._block(image, f"{prefix}.1")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return classifier logits for one normalised CHW image."""
        if image.shape != (3, 224, 224):
            raise ValueError(f"Expected an image shaped (3, 224, 224), got {image.shape}")

        output = self._conv2d(image, self._weight("conv1.weight"), stride=2, padding=3)
        output = self._normalise_batch_norm(output, "bn1")
        output = self._relu(output)
        output = self._max_pool(output)
        output = self._layer(output, 1, 1)
        output = self._layer(output, 2, 2)
        output = self._layer(output, 3, 2)
        output = self._layer(output, 4, 2)
        output = output.mean(axis=(1, 2))
        output = output @ self._weight("fc.0.weight").T + self._weight("fc.0.bias")
        output = self._relu(output)
        return output @ self._weight("fc.3.weight").T + self._weight("fc.3.bias")
