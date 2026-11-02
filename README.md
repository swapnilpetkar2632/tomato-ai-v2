# 🌿 Conformal Leaf Disease Detection & Uncertainty Quantification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Conformal Inference](https://img.shields.io/badge/Conformal_Prediction-calibrated_sets-2563eb?style=flat)](https://arxiv.org/abs/2107.07561)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computer vision and **split conformal prediction** web application for tomato leaf disease classification. Built with PyTorch (ResNet18) and Flask, it displays model scores and candidate sets formed from a calibration threshold. The 75% value is a calibration target for **marginal, in-distribution coverage under exchangeability assumptions**; it is not a guarantee for an individual image, a particular class, or arbitrary out-of-distribution input. The checked-in evaluation reports the latest empirical coverage on 500 sampled test images.

---

## 📸 Key Features

* **📊 Calibrated Candidate Sets:** Uses the saved split-conformal threshold and reports its target separately from the latest empirical test evaluation.
* **🧠 ResNet18 Transfer Learning:** Fine-tuned convolutional network with custom dense classification head.
* **🌐 Protected workspace:** Login-gated Home, Analyze, Disease Library, History, and Profile pages with a responsive mobile-first UI.
* **⚠️ Score Heuristics:** Shows softmax scores and labels the 20% top-score screen as an unvalidated heuristic, not an out-of-distribution detector.
* **🔐 Private scan history:** Successful analyses are saved automatically to the signed-in user's private SQLite history.

---

## 🏗️ Architecture & Pipeline

```
[ Raw Leaf Dataset ]
        │
        ▼
[ split_data.py ] ── Stratified 70/15/15 Split ──┐
        │                                        │
        ├──► Train Set (70%) ──► [ train.py ] ───┼──► model.pth & class_mapping.json
        ├──► Calib Set (15%) ──► [ calibrate.py ]┼──► threshold.json (q_hat at 75% target)
        └──► Test Set  (15%) ──► [ evaluate.py ] ┘──► Empirical Coverage Audit Report
                                       │
                                       ▼
                     [ predict.py & app.py (Flask) ]
                                       │
                                       ▼
                       [ Web Dashboard (http://localhost:5000) ]

[ Unlabeled OOD Images, e.g. D:\ood ] ──► [ evaluate_ood.py ] ──► ood_results.txt
```

---

## 📊 Conformal Prediction Theory

Standard softmax outputs $\hat{\pi}(y|x)$ can be dangerously overconfident on ambiguous samples. Conformal prediction addresses this by constructing a prediction set $\mathcal{C}(x)$ of candidate classes:

$$\mathcal{C}(x) = \{ y \in \mathcal{Y} \mid 1 - \hat{\pi}(y|x) \le \hat{q} \}$$

Where the quantile threshold $\hat{q}$ is calibrated on a held-out calibration set $S_{\text{calib}}$ of size $n$:

$$\hat{q} = \text{Quantile}\left(S_1, \dots, S_n; \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)$$

For target error rate $\alpha = 0.25$, split conformal targets at least 75% **marginal coverage** when calibration and future in-distribution examples are exchangeable and the procedure's assumptions hold. This does not give per-image or per-class guarantees, and it does not apply to arbitrary out-of-distribution inputs. An empty set remains empty; inference does not add heuristic classes to it.

The saved threshold records a 75% calibration target. The checked-in `results.txt` records the latest empirical coverage on its sampled test set. These are separate measurements; the empirical result is not a guarantee for future data.

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and install required Python packages:

```bash
git clone https://github.com/swapnilpetkar2632/-Tomato-Leaf-AI.git
cd -Tomato-Leaf-AI

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Rebuild the ignored model file from the Git-safe model chunks
python scripts/assemble_model.py
```

### 2. Pipeline Execution

```bash
# Step 1: Verify Environment Setup
python verify_env.py

# Step 2: Verify the project's data split, or split raw class folders in place
python split_data.py
# Optional: use another dataset root (creates train/calib/test under that path)
python split_data.py --src "C:\path\to\tomato_dataset"

# Step 3: Train ResNet18 Model
python train.py

# Step 4: Calibrate the conformal threshold (75% marginal in-distribution target)
python calibrate.py

# Step 5: Evaluate Empirical Coverage on Unseen Test Split
python evaluate.py

# Optional: audit unlabeled OOD images; this does not validate OOD detection
python evaluate_ood.py --src "D:\ood"

# Step 6: Launch Local Web Interface
python app.py
```

Access the web app by navigating to **`http://127.0.0.1:5000`** in your web browser.

### 3. Authentication, pages, and local security configuration

The application workspace requires authentication. Only `/login`, `/signup`, `/forgot-password`, `/reset-password/<token>`, static files, and `/health` are public. After signing in, the page map is:

* `/` — Home and project explanation
* `/analyze` — Protected image analysis tool
* `/diseases` and `/diseases/<slug>` — Local ten-condition disease library
* `/history` — Private automatically saved analyses with filters and pagination
* `/profile` — Name, language preference, password, and account deletion

The existing `/api/predict` request and response contract is unchanged, but logged-out requests now return JSON `{"error": "Please log in to continue."}` with status 401. The front end redirects the user to `/login` when needed.

For a persistent secret key, set it before starting Flask. A random temporary key is used only as a development fallback, so sessions and reset links will not survive a restart in that mode. When using the local HTTP address with a persistent key, keep Flask in development mode; production mode marks the session cookie `Secure` and expects HTTPS:

```powershell
$env:SECRET_KEY = "replace-with-a-long-random-secret"
$env:FLASK_DEBUG = "1"  # local HTTP only
python app.py
```

The SQLite account database and saved scan thumbnails are created under `instance/`. Passwords are stored only as Werkzeug hashes. Forgot-password responses are intentionally generic, and reset links are printed to the development log because no email provider is configured. Use HTTPS and a persistent production database/storage backend before deploying publicly. Flask-Limiter uses in-memory storage for local development; configure a shared storage URI for multi-process production deployments. Security headers include frame protection, MIME sniffing protection, referrer policy, and a permissions policy.

To run the automated checks:

```bash
python -m unittest discover -s tests -v
```

`split_data.py --src PATH` expects an existing directory containing one folder per expected tomato class. It validates the directory, all expected classes, and supported image files before copying images into `PATH/train`, `PATH/calib`, and `PATH/test`. The default source remains this project's `data` directory. If all three split directories already exist, the script reports their counts; if only some exist, it stops without copying.

The workflow requires all ten expected tomato class folders and at least one supported image in every train/calib/test class directory. Training derives the output layer size and class indices from `ImageFolder`, then verifies that the expected classes are present. Calibration, evaluation, and inference check their dataset/artifact class mappings against the saved `class_mapping.json`; mismatches fail with a clear error. Calibration and evaluation use deterministic class-stratified samples when limiting work to 500 images.

The training script requests pretrained ResNet18 ImageNet weights, which may download on the first run. **Training overwrites `model.pth` and `class_mapping.json`; calibration overwrites `threshold.json`; evaluation overwrites `results.txt`.** The app-only workflow does not run these steps.

This public app release intentionally omits the large `data/train`, `data/calib`, and `data/test` image folders. The checked-in model artifacts are sufficient for web inference. Restore a compatible dataset before running training, calibration, evaluation, or the disease-asset preparation script.

The Home page shows the calibration target separately from the most recently recorded empirical evaluation. The Analyze page uses plain-language labels such as “Model confidence” and “Possible diseases,” while its probability drawer shows only the highest and second-highest ranked classes for a focused readout. The API still returns the complete confidence list for compatibility. An empty prediction set is never described as “no disease” or “healthy.” Its top-softmax 20% screen is only a heuristic and has not been validated to detect non-leaf or out-of-distribution images. `evaluate_ood.py` produces a separate exploratory report for an unlabeled directory such as `D:\ood`; its output is not coverage evaluation or proof of OOD detection.

---

## 📁 Repository Structure

```
├── app.py                # Flask web server & API endpoints
├── auth.py               # Account, history, profile, and reset routes
├── extensions.py         # Flask-Login, SQLAlchemy, CSRF, and rate limiting
├── models.py             # User and saved-scan database models
├── predict.py            # Inference engine with conformal set logic
├── conformal_sets.py     # Shared prediction-set rule
├── class_mapping.py      # Expected-class and mapping checks
├── train.py              # ResNet18 transfer learning script
├── calibrate.py          # Conformal quantile calibration (75% target)
├── evaluate.py           # Empirical test coverage evaluation
├── evaluate_ood.py       # Exploratory unlabeled OOD behavior audit
├── split_data.py         # Stratified dataset splitter
├── paths.py              # Project-relative paths and image extensions
├── dataset_validation.py # Dataset structure and image validation
├── sampling.py           # Deterministic class-stratified sampling
├── artifact_io.py        # Atomic artifact writes
├── verify_env.py         # Hardware & dependency validator
├── templates/
│   ├── base.html         # Shared navigation, flash messages, and CSRF token
│   ├── index.html        # Protected Home page
│   ├── analyze.html      # Protected analysis UI and unchanged API client
│   ├── diseases.html     # Disease library grid and filters
│   ├── disease_detail.html # Disease detail page
│   ├── error.html         # Friendly HTML error states
│   ├── login.html        # Sign-in page
│   ├── signup.html       # Account creation page
│   ├── forgot_password.html
│   ├── reset_password.html
│   ├── history.html      # Saved scan history
│   └── profile.html      # Profile, password, language, and account deletion
├── data/diseases.json    # Ten cautious disease reference profiles
├── static/img/diseases/  # Local resized sample images
├── scripts/prepare_disease_assets.py # Rebuilds local library thumbnails
├── class_mapping.json    # Target class index mapping
├── threshold.json       # Calibrated quantile q_hat metadata
├── requirements.txt      # Python package dependencies
├── tests/                # Runtime and authentication contract tests
├── README.md             # Project documentation
└── .gitignore            # Git exclusion settings
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
