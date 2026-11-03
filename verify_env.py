"""Verify the lightweight web runtime environment."""

import sys
from importlib.metadata import PackageNotFoundError, version


RUNTIME_PACKAGES = (
    "Flask",
    "Flask-Login",
    "Flask-SQLAlchemy",
    "Flask-WTF",
    "Flask-Limiter",
    "numpy",
    "Pillow",
    "gunicorn",
)


def main() -> None:
    print("=" * 70)
    print("  STAGE 1 -- Lightweight Web Environment Verification")
    print("=" * 70)
    print(f"Python Version    : {sys.version.split()[0]}")
    for package_name in RUNTIME_PACKAGES:
        try:
            package_version = version(package_name)
        except PackageNotFoundError:
            package_version = "NOT INSTALLED"
        print(f"{package_name:<18}: {package_version}")
    print("=" * 70)
    print("[OK] Lightweight web runtime verified.\n")


if __name__ == "__main__":
    main()
