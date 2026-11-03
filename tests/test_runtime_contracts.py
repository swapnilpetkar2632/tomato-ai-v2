import json
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import predict
from app import app
from class_mapping import EXPECTED_CLASSES, ensure_same_class_mapping, validate_class_mapping
from conformal_sets import prediction_set_indices
from evaluate_ood import collect_images
from extensions import db
from models import Scan, User
from paths import CLASS_MAP_PATH, DATA_TEST, MODEL_NPZ_PATH, MODEL_PATH, THRESH_PATH
from sampling import stratified_sample_indices
import split_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FlatModel:
    def predict(self, image):
        logits = np.zeros(10, dtype=np.float32)
        logits[0] = np.log(3.0)
        return logits


class RuntimeContracts(unittest.TestCase):
    runtime_email = "runtime-contract@example.test"
    runtime_password = "Runtime-Horse-42"

    def _cleanup_runtime_user(self):
        with app.app_context():
            user = User.query.filter_by(email=self.runtime_email).first()
            if user:
                Scan.query.filter_by(user_id=user.id).delete()
                db.session.delete(user)
                db.session.commit()

    def tearDown(self):
        self._cleanup_runtime_user()

    def authenticated_client(self, ip):
        self._cleanup_runtime_user()
        client = app.test_client()
        page = client.get("/signup", environ_base={"REMOTE_ADDR": ip})
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True)).group(1)
        response = client.post(
            "/signup",
            data={
                "csrf_token": token,
                "name": "Runtime Test User",
                "email": self.runtime_email,
                "password": self.runtime_password,
                "confirm_password": self.runtime_password,
            },
            environ_base={"REMOTE_ADDR": ip},
        )
        self.assertEqual(response.status_code, 302)
        return client

    def test_class_mapping_requires_expected_classes_and_contiguous_indices(self):
        mapping = {name: index for index, name in enumerate(sorted(EXPECTED_CLASSES))}
        self.assertEqual(validate_class_mapping(mapping, "test data"), mapping)

        with self.assertRaisesRegex(ValueError, "missing expected tomato classes"):
            validate_class_mapping({"Tomato_healthy": 0}, "test data")

        mapping["Tomato_healthy"] = len(mapping) + 1
        with self.assertRaisesRegex(ValueError, "unique and start at 0"):
            validate_class_mapping(mapping, "test data")

    def test_class_mapping_comparison_rejects_index_drift(self):
        reference = {name: index for index, name in enumerate(sorted(EXPECTED_CLASSES))}
        changed = dict(reference)
        changed["Tomato_healthy"], changed["Tomato_mosaic_virus"] = (
            changed["Tomato_mosaic_virus"], changed["Tomato_healthy"]
        )

        with self.assertRaisesRegex(ValueError, "does not match the reference"):
            ensure_same_class_mapping(reference, changed, "calibration data")

    def test_shared_conformal_rule_keeps_empty_sets_empty(self):
        self.assertEqual(prediction_set_indices([0.25, 0.75], q_hat=0.01), [])
        self.assertEqual(prediction_set_indices([0.995, 0.005], q_hat=0.01), [0])

    def test_prediction_preserves_an_empty_conformal_set(self):
        with (PROJECT_ROOT / "class_mapping.json").open() as mapping_file:
            class_mapping = json.load(mapping_file)
        threshold = {"q_hat": 0.01, "class_mapping": class_mapping}

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "leaf.png"
            Image.new("RGB", (32, 32), "green").save(image_path)
            with patch(
                "predict._get_inference_engine",
                return_value=(FlatModel(), threshold),
            ):
                result = predict.predict_with_uncertainty(image_path)

        self.assertEqual(result["prediction_set"], [])
        self.assertEqual(result["prediction_set_status"], "empty")

    def test_project_artifact_paths_are_independent_of_working_directory(self):
        self.assertEqual(MODEL_PATH, PROJECT_ROOT / "model.pth")
        self.assertEqual(MODEL_NPZ_PATH, PROJECT_ROOT / "model.npz")
        self.assertEqual(THRESH_PATH, PROJECT_ROOT / "threshold.json")
        self.assertEqual(CLASS_MAP_PATH, PROJECT_ROOT / "class_mapping.json")
        self.assertEqual(DATA_TEST, PROJECT_ROOT / "data" / "test")

    def test_sample_discovery_supports_png_from_any_working_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "nested" / "leaf.PNG"
            image_path.parent.mkdir()
            Image.new("RGB", (4, 4), "green").save(image_path)

            original_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                self.assertEqual(predict.discover_sample_images(Path(temp_dir)), [image_path])
            finally:
                os.chdir(original_cwd)

    def test_stratified_sampling_is_deterministic_and_balanced(self):
        targets = [0] * 6 + [1] * 6 + [2] * 6
        first = stratified_sample_indices(targets, max_samples=9)
        second = stratified_sample_indices(targets, max_samples=9)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 9)
        self.assertEqual([sum(targets[index] == label for index in first) for label in range(3)], [3, 3, 3])

    def test_upload_rejects_invalid_image_with_bad_request(self):
        response = self.authenticated_client("10.20.0.1").post(
            "/api/predict",
            data={"image": (BytesIO(b"not an image"), "leaf.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_index_and_missing_upload_statuses(self):
        client = self.authenticated_client("10.20.0.2")

        index = client.get("/")
        self.assertEqual(index.status_code, 200)
        self.assertIn(b"target coverage", index.data)
        response = client.post("/api/predict")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

        sample_path_response = client.post(
            "/api/predict", data={"sample_path": "../../model.pth"}
        )
        self.assertEqual(sample_path_response.status_code, 400)

    def test_ui_prompts_for_a_clear_image_below_75_percent_confidence(self):
        template = (PROJECT_ROOT / "templates" / "analyze.html").read_text(encoding="utf-8")
        self.assertIn("const lowConfidenceThreshold = 75;", template)
        self.assertIn("Please upload a clear, well-lit image showing a single tomato leaf.", template)
        self.assertIn("const topTwoConfidences = (data.confidences || []).slice(0, 2);", template)
        self.assertIn("Top 2 class probabilities", template)

    def test_internal_prediction_errors_do_not_leak_details(self):
        image_bytes = BytesIO()
        Image.new("RGB", (8, 8), "green").save(image_bytes, format="PNG")
        image_bytes.seek(0)
        with patch("app.predict_with_uncertainty", side_effect=RuntimeError("private detail")):
            with self.assertLogs(app.logger, level="ERROR"):
                response = self.authenticated_client("10.20.0.3").post(
                    "/api/predict",
                    data={"image": (image_bytes, "leaf.png")},
                    content_type="multipart/form-data",
                )

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private detail", response.get_json()["error"])

    def test_splitter_help_documents_source_option(self):
        result = subprocess.run(
            [sys.executable, "split_data.py", "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--src", result.stdout)

    def test_source_root_is_validated_before_splitting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            for class_name in EXPECTED_CLASSES:
                class_dir = source / class_name
                class_dir.mkdir()
                for index in range(10):
                    Image.new("RGB", (4, 4), "green").save(class_dir / f"{index}.png")

            with contextlib.redirect_stdout(io.StringIO()):
                split_data.organize_and_split(source)

            for split_name in ("train", "calib", "test"):
                for class_name in EXPECTED_CLASSES:
                    self.assertTrue((source / split_name / class_name).is_dir())

    def test_missing_source_classes_fail_before_creating_split_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            class_dir = source / "Tomato_healthy"
            class_dir.mkdir()
            Image.new("RGB", (4, 4), "green").save(class_dir / "leaf.png")

            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "missing expected tomato classes"):
                    split_data.organize_and_split(source)

            self.assertFalse((source / "train").exists())

    def test_small_raw_classes_fail_before_creating_split_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            for class_name in EXPECTED_CLASSES:
                class_dir = source / class_name
                class_dir.mkdir()
                for index in range(6):
                    Image.new("RGB", (4, 4), "green").save(class_dir / f"{index}.png")

            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "enough images"):
                    split_data.organize_and_split(source)

            self.assertFalse((source / "train").exists())

    def test_existing_incomplete_split_fails_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            for split_name in ("train", "calib", "test"):
                for class_name in EXPECTED_CLASSES[:-1]:
                    class_dir = source / split_name / class_name
                    class_dir.mkdir(parents=True)
                    Image.new("RGB", (4, 4), "green").save(class_dir / "leaf.png")

            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "missing expected tomato classes"):
                    split_data.organize_and_split(source)

    def test_ood_image_collection_is_recursive_and_extension_aware(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            image_path = source / "nested" / "leaf.JPEG"
            image_path.parent.mkdir()
            Image.new("RGB", (4, 4), "green").save(image_path)

            self.assertEqual(collect_images(source), [image_path.resolve()])


if __name__ == "__main__":
    unittest.main()
