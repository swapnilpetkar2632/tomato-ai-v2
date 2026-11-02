# Project Plan: Conformal Leaf Disease Detection & Uncertainty Quantification

## Executive Summary
This project delivers an end-to-end computer vision and conformal inference web application for tomato leaf classification. It uses a calibrated split-conformal threshold to form candidate sets and also shows raw model scores. The 75% value is a target for marginal in-distribution coverage under exchangeability assumptions, not a per-image, per-class, legal, or arbitrary-OOD guarantee. The checked-in test report records the latest empirical coverage on 500 sampled images.

---

## 🏗️ System Architecture & Workflow

```
[Raw / Custom Dataset]
       │
       ▼
[split_data.py] ─── Stratified 70/15/15 Split ───┐
       │                                         │
       ├──► Train Set (70%) ──► [train.py] ──────┼──► model.pth & class_mapping.json
       ├──► Calib Set (15%) ──► [calibrate.py] ──┼──► threshold.json (q_hat at 75% target)
       └──► Test Set  (15%) ──► [evaluate.py] ───┘──► Empirical Coverage Audit Report
                                       │
                                       ▼
                     [predict.py & app.py (Flask Server)]
                                       │
                                       ▼
                       [Light Blue Web Interface]

[Unlabeled OOD Images, e.g. D:\ood] ──► [evaluate_ood.py] ──► Exploratory Audit
```

---

## 📑 Multi-Stage Project Roadmap

### Stage 1: Environment Setup Verification
* **Script:** `verify_env.py`
* **Purpose:** Validates Flask, PyTorch, Torchvision, NumPy, Scikit-Learn, Pillow, tqdm, and CUDA/CPU hardware acceleration.

### Stage 2: Data Organization & Stratified Splitting
* **Script:** `split_data.py`
* **Purpose:** Inspects raw leaf image directories, maps folder names to the ten expected tomato classes, and creates a 70% Train / 15% Calibration / 15% Test split.
* **CLI Feature:** `--src <path>` selects an existing raw dataset root; the split folders are created under that same path. With no option, the project `data` directory is used. The source, expected class folders, supported images, and non-empty train/calib/test class splits are validated before copying or reporting success. Partial or incomplete split directories stop with an error.

### Stage 3: Transfer Learning & Model Training
* **Script:** `train.py`
* **Architecture:** ResNet18 initialized with pretrained ImageNet weights; layer3 and layer4 are fine-tuned with a 256-unit dense head, ReLU, and dropout. The output size and class indices come from `ImageFolder`; training fails if expected tomato classes are missing or unexpected classes are present.
* **Optimization:** CrossEntropyLoss, Adam optimizer ($lr=5\times 10^{-4}$), ReduceLROnPlateau scheduler.
* **Artifacts Generated:** `model.pth`, `class_mapping.json`.
* **Compatibility:** Calibration, evaluation, and inference compare their dataset or threshold mapping with `class_mapping.json` before loading the checkpoint.
* **Download:** The first run may download pretrained ImageNet weights.

### Stage 4: Conformal Uncertainty Calibration
* **Script:** `calibrate.py`
* **Target Coverage:** $1 - \alpha = 75\%$ ($\alpha = 0.25$), a marginal in-distribution target under exchangeability assumptions.
* **Methodology:** 
  1. Computes non-conformity scores on calibration samples: $S_i = 1 - \hat{\pi}(y_i | x_i)$.
  2. Uses a deterministic class-stratified sample of up to 500 calibration images.
  3. Calculates non-parametric quantile level $\hat{q} = \text{Quantile}\left(S_1, \dots, S_n; \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)$.
* **Artifact Generated:** `threshold.json`.

### Stage 5: Empirical Evaluation
* **Script:** `evaluate.py`
* **Purpose:** Reports empirical test coverage and candidate-set sizes; this is a sample measurement, not proof that a future or per-class result reaches 75%.
* **Checked-in result:** `results.txt` reports 77.80% (389/500) on its 500-image sample.
* **Sampling note:** The evaluation sample is deterministic and class-stratified when the test set exceeds 500 images; this supports a more informative per-class audit but remains a sample measurement.
* **Public app release:** The large `data/train`, `data/calib`, and `data/test` image folders are omitted after artifact generation. Restore a compatible dataset before rerunning the ML pipeline.

### Stage 6: Exploratory OOD Behavior Audit
* **Script:** `evaluate_ood.py`
* **Usage:** `python evaluate_ood.py --src "D:\ood"` audits supported images recursively without labels.
* **Output:** `ood_results.txt` reports softmax-score and prediction-set behavior separately from conformal test coverage.
* **Limitation:** This is exploratory only. The 20% top-softmax screen remains an unvalidated heuristic and is not an OOD detector or guarantee.

### Stage 7: Inference Engine & Preprocessing
* **Script:** `predict.py`
* **Feature:** Inference transforms (`Resize(256)` $\to$ `CenterCrop(224)`), softmax score computation, and the same candidate-set rule as evaluation: $\mathcal{C}(x) = \{ y : 1 - \hat{\pi}(y|x) \le \hat{q} \}$. Empty sets remain empty; inference does not substitute heuristic classes.
* **OOD limitation:** The top-softmax score threshold of 0.20 is an unvalidated display heuristic, not a reliable non-leaf or OOD detector.

### Stage 8: Web Application & UI System
* **Script:** `app.py`, `templates/index.html`
* **Design System:** Light Blue theme (`#f0f7ff` background, `#2563eb` primary palette, white card shadow components).
* **Interactions:** Drag-and-drop upload, image preview/reset, candidate set, softmax scores, and a clear note that the leaf-screen heuristic is unvalidated. Invalid uploads receive HTTP 4xx responses; internal prediction failures do not expose tracebacks.

---

## 📊 Mathematical Foundation (Conformal Prediction)

Under exchangeability of calibration and future in-distribution examples, split conformal targets marginal coverage over future samples:

$$P(Y_{n+1} \in \mathcal{C}(X_{n+1})) \ge 1 - \alpha$$

where the prediction set is constructed as:

$$\mathcal{C}(x) = \{ y \in \mathcal{Y} \mid 1 - \hat{f}(x)_y \le \hat{q} \}$$

This is not a guarantee for an individual image or class and does not apply to arbitrary OOD inputs. The current checked-in empirical result is 77.80% on 500 sampled test images.

---

## 🚀 Execution Commands Guide

```powershell
# 1. Activate Environment
.\venv\Scripts\activate

# 2. Verify Setup
python verify_env.py

# 3. Verify/split the project data, or split a raw dataset root in place:
# python split_data.py --src "C:\path\to\tomato_dataset"
python split_data.py

# 4. Train ResNet18 Model
python train.py

# 5. Calibrate the conformal threshold (75% marginal in-distribution target)
python calibrate.py

# 6. Evaluate Test Coverage
python evaluate.py

# Optional: audit unlabeled OOD images without changing test results:
python evaluate_ood.py --src "D:\ood"

# 7. Launch Web Application
python app.py
```

`Flask` is installed from `requirements.txt`. Training may download ImageNet weights and overwrites `model.pth` and `class_mapping.json`; calibration overwrites `threshold.json`; evaluation overwrites `results.txt`. The web app does not run these pipeline stages.
