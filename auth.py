"""Account, scan-history, and profile routes for the protected workspace."""

import json
import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    send_file,
    url_for,
)
from flask_login import current_user, login_required, login_user, logout_user
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from PIL import Image
from disease_data import library_link
from extensions import db, limiter
from models import Scan, User
import storage


auth = Blueprint("auth", __name__)
SUPPORTED_LANGUAGES = {"en", "hi", "mr", "bn"}
PASSWORD_MIN_LENGTH = 8


def _normalize_email(value: str) -> str:
    return value.strip().lower()


def _safe_next_url(value: str | None) -> bool:
    if not value:
        return True
    parsed = urlsplit(value)
    return (
        value.startswith("/")
        and not value.startswith("//")
        and "\\" not in value
        and not parsed.scheme
        and not parsed.netloc
    )


def _next_or_home(value: str | None) -> str:
    return value if _safe_next_url(value) and value else url_for("index")


def _reset_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"], salt="tomato-leaf-password-reset")


def _thumbnail_directory() -> Path:
    directory = Path(current_app.instance_path) / "scan_thumbnails"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _remove_thumbnail(scan: Scan) -> None:
    if not scan.thumbnail_path:
        return
    if storage.configured():
        try:
            storage.delete(scan.thumbnail_path)
        except Exception:
            current_app.logger.warning("Could not remove stored thumbnail %s", scan.thumbnail_path)
        return
    (_thumbnail_directory() / Path(scan.thumbnail_path).name).unlink(missing_ok=True)


@auth.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    next_url = request.args.get("next") or request.form.get("next")
    if request.method == "POST":
        email = _normalize_email(request.form.get("email", ""))
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash("Email or password is incorrect.", "error")
        else:
            login_user(user, remember=request.form.get("remember") == "on")
            flash("Welcome back.", "success")
            return redirect(_next_or_home(next_url))

    return render_template("login.html", next=next_url if _safe_next_url(next_url) else "")


@auth.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    form_values = {"name": "", "email": ""}
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = _normalize_email(request.form.get("email", ""))
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        form_values = {"name": name, "email": email}

        if not name or len(name) > 120:
            flash("Please enter your name.", "error")
        elif "@" not in email or len(email) > 320:
            flash("Please enter a valid email address.", "error")
        elif len(password) < PASSWORD_MIN_LENGTH:
            flash(f"Password must be at least {PASSWORD_MIN_LENGTH} characters.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        elif User.query.filter_by(email=email).first():
            flash("Unable to create an account with those details.", "error")
        else:
            user = User(name=name, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash("Your account has been created.", "success")
            return redirect(url_for("index"))

    return render_template("signup.html", form_values=form_values)


@auth.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    message = None
    if request.method == "POST":
        email = _normalize_email(request.form.get("email", ""))
        user = User.query.filter_by(email=email).first()
        if user:
            token = _reset_serializer().dumps({"user_id": user.id})
            reset_link = url_for("auth.reset_password", token=token, _external=True)
            current_app.logger.info("Development password reset link: %s", reset_link)
            print(f"Development password reset link: {reset_link}")
        message = "If an account exists for that email, a reset link has been printed to the development console."

    return render_template("forgot_password.html", message=message)


@auth.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token: str):
    try:
        payload = _reset_serializer().loads(token, max_age=3600)
        user = db.session.get(User, int(payload["user_id"]))
    except (BadSignature, SignatureExpired, KeyError, TypeError, ValueError):
        user = None

    if user is None:
        flash("This password reset link is invalid or expired.", "error")
        return redirect(url_for("auth.forgot_password"))

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        if len(password) < PASSWORD_MIN_LENGTH:
            flash(f"Password must be at least {PASSWORD_MIN_LENGTH} characters.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        else:
            user.set_password(password)
            db.session.commit()
            flash("Your password has been reset. You can now log in.", "success")
            return redirect(url_for("auth.login"))

    return render_template("reset_password.html", token=token)


@auth.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("index"))


@auth.route("/history")
@login_required
def history():
    selected_disease = request.args.get("disease", "").strip()
    selected_date = request.args.get("date", "").strip()
    page = max(request.args.get("page", 1, type=int), 1)
    query = Scan.query.filter_by(user_id=current_user.id)
    if selected_disease:
        query = query.filter(Scan.top_disease == selected_disease)
    if selected_date:
        try:
            day = datetime.strptime(selected_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            query = query.filter(
                Scan.created_at >= day,
                Scan.created_at < day + timedelta(days=1),
            )
        except ValueError:
            selected_date = ""

    pagination = query.order_by(Scan.created_at.desc()).paginate(page=page, per_page=8, error_out=False)
    available_diseases = [
        row[0]
        for row in Scan.query.with_entities(Scan.top_disease)
        .filter_by(user_id=current_user.id)
        .distinct()
        .order_by(Scan.top_disease)
        .all()
    ]
    return render_template(
        "history.html",
        scans=pagination.items,
        pagination=pagination,
        selected_disease=selected_disease,
        selected_date=selected_date,
        available_diseases=available_diseases,
        library_link=library_link,
    )


@auth.route("/history/delete/<int:scan_id>", methods=["POST"])
@login_required
def delete_scan(scan_id: int):
    scan = Scan.query.filter_by(id=scan_id, user_id=current_user.id).first_or_404()
    _remove_thumbnail(scan)
    db.session.delete(scan)
    db.session.commit()
    flash("Scan deleted.", "success")
    return redirect(url_for("auth.history"))


@auth.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    scans = Scan.query.filter_by(user_id=current_user.id).all()
    for scan in scans:
        _remove_thumbnail(scan)
        db.session.delete(scan)
    db.session.commit()
    flash("Your scan history has been cleared.", "success")
    return redirect(url_for("auth.history"))


@auth.route("/history/thumbnail/<int:scan_id>")
@login_required
def scan_thumbnail(scan_id: int):
    scan = Scan.query.filter_by(id=scan_id, user_id=current_user.id).first_or_404()
    if not scan.thumbnail_path:
        abort(404)
    if storage.configured():
        try:
            return send_file(BytesIO(storage.get(scan.thumbnail_path)), mimetype="image/jpeg", max_age=31536000)
        except FileNotFoundError:
            abort(404)
    return send_from_directory(str(_thumbnail_directory()), Path(scan.thumbnail_path).name)


@auth.route("/api/history/save", methods=["POST"])
@login_required
def save_history():
    image_file = request.files.get("image")
    raw_result = request.form.get("result", "")
    thumbnail_key: str | None = None
    local_thumbnail_path: Path | None = None
    if image_file is None or not image_file.filename:
        return jsonify({"error": "A scan image is required."}), 400

    try:
        result = json.loads(raw_result)
        prediction_set = result.get("prediction_set", [])
        if not isinstance(prediction_set, list) or not all(isinstance(item, str) for item in prediction_set):
            raise ValueError("Invalid prediction set")
        top_disease = str(result["top_disease"])[:200]
        top_confidence = float(result["top_confidence_percentage"]) / 100.0
        set_size = int(result["set_size"])
        if not top_disease or not 0.0 <= top_confidence <= 1.0 or set_size < 0:
            raise ValueError("Invalid confidence data")

        image_file.stream.seek(0)
        with Image.open(image_file.stream) as image:
            image.verify()
        image_file.stream.seek(0)
        with Image.open(image_file.stream) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((360, 260))
            filename = f"{uuid.uuid4().hex}.jpg"
            thumbnail_key = f"scan-thumbnails/{filename}"
            thumbnail_bytes = BytesIO()
            thumbnail.save(thumbnail_bytes, format="JPEG", quality=86, optimize=True)
        if storage.configured():
            storage.put(thumbnail_key, thumbnail_bytes.getvalue())
        else:
            local_thumbnail_path = _thumbnail_directory() / filename
            local_thumbnail_path.write_bytes(thumbnail_bytes.getvalue())

        scan = Scan(
            user_id=current_user.id,
            top_disease=top_disease,
            top_confidence=top_confidence,
            prediction_set=prediction_set,
            set_size=set_size,
            thumbnail_path=thumbnail_key,
        )
        db.session.add(scan)
        db.session.commit()
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError):
        db.session.rollback()
        if thumbnail_key and storage.configured():
            try:
                storage.delete(thumbnail_key)
            except Exception:
                current_app.logger.warning("Could not remove failed thumbnail %s", thumbnail_key)
        if local_thumbnail_path is not None:
            local_thumbnail_path.unlink(missing_ok=True)
        return jsonify({"error": "The scan could not be saved."}), 400

    return jsonify({"success": True, "scan_id": scan.id}), 201


@auth.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "profile":
            name = request.form.get("name", "").strip()
            language = request.form.get("language", "en")
            if not name or len(name) > 120:
                flash("Please enter your name.", "error")
            elif language not in SUPPORTED_LANGUAGES:
                flash("Please choose a supported language.", "error")
            else:
                current_user.name = name
                current_user.language = language
                db.session.commit()
                flash("Profile updated.", "success")
        elif action == "password":
            current_password = request.form.get("current_password", "")
            new_password = request.form.get("new_password", "")
            confirm_password = request.form.get("confirm_password", "")
            if not current_user.check_password(current_password):
                flash("Current password is incorrect.", "error")
            elif len(new_password) < PASSWORD_MIN_LENGTH:
                flash(f"Password must be at least {PASSWORD_MIN_LENGTH} characters.", "error")
            elif new_password != confirm_password:
                flash("Passwords do not match.", "error")
            else:
                current_user.set_password(new_password)
                db.session.commit()
                flash("Password changed.", "success")
        elif action == "delete":
            if request.form.get("confirmation") != "DELETE":
                flash("Type DELETE to confirm account removal.", "error")
            else:
                for scan in list(current_user.scans):
                    _remove_thumbnail(scan)
                db.session.delete(current_user)
                db.session.commit()
                logout_user()
                flash("Your account has been deleted.", "success")
                return redirect(url_for("index"))

    return render_template("profile.html", languages=sorted(SUPPORTED_LANGUAGES))
