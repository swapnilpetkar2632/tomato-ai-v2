"""Flask web application for the protected tomato-leaf assistant."""

import json
import os
import re
import secrets
import tempfile
import warnings
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for
from flask_login import current_user
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge

from auth import auth
from disease_data import disease_items, find_disease, slugify
from extensions import csrf, db, limiter, login_manager
from models import User
from paths import RESULTS_PATH, THRESH_PATH
from predict import predict_with_uncertainty


app = Flask(__name__)
secret_key = os.environ.get("SECRET_KEY")
debug_enabled = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
development_mode = not secret_key or debug_enabled
if not secret_key:
    warnings.warn(
        "SECRET_KEY is not set; using a temporary development key. Set SECRET_KEY for persistent sessions.",
        RuntimeWarning,
        stacklevel=2,
    )
    secret_key = secrets.token_urlsafe(32)

database_url = os.environ.get("DATABASE_URL", "").strip()
if database_url.startswith("postgres://"):
    database_url = "postgresql+psycopg://" + database_url.removeprefix("postgres://")
elif database_url.startswith("postgresql://"):
    database_url = "postgresql+psycopg://" + database_url.removeprefix("postgresql://")

database_path = Path(app.instance_path) / "app.db"
app.config.from_mapping(
    SECRET_KEY=secret_key,
    DEBUG=debug_enabled,
    SQLALCHEMY_DATABASE_URI=database_url or f"sqlite:///{database_path.as_posix()}",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    TEMPLATES_AUTO_RELOAD=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=not development_mode,
)
if database_url.startswith("postgresql+"):
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }

Path(app.instance_path).mkdir(parents=True, exist_ok=True)
db.init_app(app)
login_manager.init_app(app)
csrf.init_app(app)
limiter.init_app(app)
app.register_blueprint(auth)


@login_manager.user_loader
def load_user(user_id: str):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None


with app.app_context():
    db.create_all()


THRESHOLD_PATH = THRESH_PATH
PUBLIC_ENDPOINTS = {
    "auth.login",
    "auth.signup",
    "auth.forgot_password",
    "auth.reset_password",
    "health",
    "static",
}


def coverage_summary() -> dict:
    """Read the calibrated target and most recently recorded test result."""
    target_coverage = None
    empirical_coverage = None
    evaluation_sample_count = None

    try:
        threshold_data = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
        target_coverage = float(threshold_data["target_coverage"]) * 100
    except (OSError, ValueError, KeyError, TypeError):
        pass

    try:
        results_text = RESULTS_PATH.read_text(encoding="utf-8")
    except OSError:
        results_text = ""
    match = re.search(r"Empirical Coverage\s*:\s*([\d.]+)%\s*\(\d+/(\d+)\)", results_text)
    if match:
        empirical_coverage = float(match.group(1))
        evaluation_sample_count = int(match.group(2))

    return {
        "target_coverage": target_coverage,
        "empirical_coverage": empirical_coverage,
        "evaluation_sample_count": evaluation_sample_count,
    }


@app.before_request
def require_login_for_application_routes():
    """Make new application routes protected by default."""
    if request.endpoint in PUBLIC_ENDPOINTS or current_user.is_authenticated:
        return None
    if request.path.startswith("/api/"):
        return jsonify({"error": "Please log in to continue."}), 401
    next_url = request.full_path.rstrip("?")
    return redirect(url_for("auth.login", next=next_url))


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
    if request.is_secure:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


@app.route("/")
def index():
    return render_template(
        "index.html",
        diseases=disease_items(),
        **coverage_summary(),
    )


@app.route("/analyze")
def analyze():
    disease_map = {}
    for disease in disease_items():
        info = {
            "slug": disease["slug"],
            "summary": disease["summary"],
            "symptom_snapshot": disease["symptom_snapshot"],
        }
        aliases = {
            disease["name"],
            disease["name"].removeprefix("Tomato "),
            disease["class_name"].replace("Tomato_", "").replace("_", " "),
            disease["class_name"],
        }
        for alias in aliases:
            disease_map[alias] = info
            disease_map[alias.lower()] = info
    return render_template("analyze.html", disease_map=disease_map, **coverage_summary())


@app.route("/diseases")
def disease_library():
    query = request.args.get("q", "").strip()
    selected_type = request.args.get("type", "all").strip().lower()
    allowed_types = {"all", "bacterial", "fungal", "viral", "pest", "healthy"}
    if selected_type not in allowed_types:
        selected_type = "all"
    diseases = disease_items()
    if query:
        query_lower = query.lower()
        diseases = [
            disease
            for disease in diseases
            if query_lower in disease["name"].lower()
            or query_lower in disease["summary"].lower()
        ]
    if selected_type != "all":
        diseases = [disease for disease in diseases if disease["type"] == selected_type]
    return render_template(
        "diseases.html",
        diseases=diseases,
        query=query,
        selected_type=selected_type,
        disease_count=len(diseases),
    )


@app.route("/diseases/<slug>")
def disease_detail(slug: str):
    disease = find_disease(slug)
    if disease is None:
        abort(404)
    return render_template("disease_detail.html", disease=disease, similar_slug=slugify)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Uploaded image exceeds the 16 MB limit."}), 413
    return render_template("error.html", title="That file is too large", message="Please choose an image smaller than 16 MB."), 413


@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "The requested resource was not found."}), 404
    return render_template("error.html", title="Page not found", message="That page is not available. Use the navigation to continue."), 404


@app.errorhandler(500)
def internal_error(_error):
    db.session.rollback()
    if request.path.startswith("/api/"):
        return jsonify({"error": "Something went wrong. Please try again."}), 500
    return render_template("error.html", title="Something went wrong", message="Please try again, or return to Home."), 500


@app.route("/api/predict", methods=["POST"])
@csrf.exempt
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "Upload an image file."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Choose an image file to upload."}), 400

    try:
        with Image.open(file.stream) as uploaded_image:
            uploaded_image.verify()
        file.stream.seek(0)
    except Exception:
        return jsonify({"error": "The uploaded file is not a valid image."}), 400

    temp_file_path = None
    try:
        suffix = Path(file.filename).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,12}", suffix):
            suffix = ".img"
        with tempfile.NamedTemporaryFile(
            prefix="tomato-leaf-", suffix=suffix, delete=False
        ) as temp_file:
            temp_file_path = Path(temp_file.name)
            file.save(temp_file)

        results = predict_with_uncertainty(temp_file_path)
        leaf_screen_passed = results.get(
            "leaf_screen_passed", results.get("is_tomato_leaf", True)
        )

        formatted_confidences = []
        for cls_name, prob in sorted(
            results["confidences"].items(), key=lambda item: item[1], reverse=True
        ):
            formatted_confidences.append(
                {
                    "class_name": cls_name.replace("Tomato_", "").replace("_", " "),
                    "raw_name": cls_name,
                    "percentage": round(prob * 100, 2),
                    "in_set": cls_name in results["prediction_set"],
                }
            )

        top_raw_class = results["top_class"]
        top_disease_clean = top_raw_class.replace("Tomato_", "").replace("_", " ")

        return jsonify(
            {
                "success": True,
                "is_tomato_leaf": leaf_screen_passed,
                "leaf_screen_passed": leaf_screen_passed,
                "leaf_screen_threshold": results.get("leaf_screen_threshold", 0.20),
                "leaf_screen_validated": results.get("leaf_screen_validated", False),
                "top_disease": top_disease_clean,
                "top_confidence_percentage": round(results["top_confidence"] * 100, 2),
                "is_confident": results["is_confident"],
                "uncertainty_level": results.get("uncertainty_level", "HIGH"),
                "prediction_set_status": results.get(
                    "prediction_set_status",
                    "nonempty" if results["prediction_set"] else "empty",
                ),
                "prediction_set": [
                    item.replace("Tomato_", "").replace("_", " ")
                    for item in results["prediction_set"]
                ],
                "raw_prediction_set": results["prediction_set"],
                "set_size": results["set_size"],
                "confidences": formatted_confidences,
                "target_coverage": results.get("target_coverage"),
            }
        )
    except Exception:
        app.logger.exception("Prediction request failed")
        return jsonify({"error": "Prediction failed. Check the server log for details."}), 500
    finally:
        if temp_file_path is not None:
            try:
                temp_file_path.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    print("=" * 70)
    print("  Starting Conformal Leaf Disease Detection Web App")
    print("  Open in browser: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "5000")),
        debug=app.config["DEBUG"],
        use_reloader=False,
    )
