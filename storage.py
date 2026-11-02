"""Small S3-compatible storage adapter for deployment-safe scan thumbnails."""

from __future__ import annotations

import os


def _first_configured(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def bucket_name() -> str | None:
    return _first_configured(
        "S3_BUCKET_NAME",
        "S3_BUCKET",
        "STORAGE_BUCKET",
        "BUCKET_NAME",
        "AWS_S3_BUCKET",
        "AWS_BUCKET_NAME",
    )


def configured() -> bool:
    """Return whether the deployment supplied an object-storage bucket."""
    return bucket_name() is not None


def _client():
    import boto3
    from botocore.config import Config

    endpoint_url = _first_configured(
        "S3_ENDPOINT_URL",
        "S3_ENDPOINT",
        "AWS_S3_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "AWS_ENDPOINT_URL",
    )
    region_name = _first_configured("AWS_REGION", "AWS_DEFAULT_REGION", "S3_REGION")
    force_path_style = _first_configured("AWS_S3_FORCE_PATH_STYLE", "S3_FORCE_PATH_STYLE")
    config = Config(s3={"addressing_style": "path"}) if force_path_style == "true" else None
    return boto3.client("s3", endpoint_url=endpoint_url, region_name=region_name, config=config)


def put(key: str, payload: bytes) -> None:
    name = bucket_name()
    if not name:
        raise RuntimeError("Object storage is not configured")
    _client().put_object(
        Bucket=name,
        Key=key,
        Body=payload,
        ContentType="image/jpeg",
        CacheControl="private, max-age=31536000",
    )


def get(key: str) -> bytes:
    name = bucket_name()
    if not name:
        raise RuntimeError("Object storage is not configured")
    try:
        return _client().get_object(Bucket=name, Key=key)["Body"].read()
    except Exception as error:
        if getattr(error, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
            raise FileNotFoundError(key) from error
        raise


def delete(key: str) -> None:
    name = bucket_name()
    if not name:
        return
    _client().delete_object(Bucket=name, Key=key)
