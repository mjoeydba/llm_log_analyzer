"""Shared helpers for persisting Elasticsearch connection settings."""

from __future__ import annotations

import base64
import getpass
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

try:  # pragma: no cover - optional dependency
    from cryptography.fernet import Fernet
except ImportError:  # pragma: no cover - optional dependency
    Fernet = None  # type: ignore[misc]


SECRET_ENV = "LLM_LOG_ANALYZER_SECRET"
logger = logging.getLogger("llm_log_analyzer.config")


def _get_cipher() -> Optional[Fernet]:
    if Fernet is None:
        return None

    secret = os.environ.get(SECRET_ENV) or getpass.getuser()
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    key = base64.urlsafe_b64encode(digest)
    return Fernet(key)


def encrypt_password(password: str) -> Dict[str, str]:
    if not password:
        return {"scheme": "none", "value": ""}

    cipher = _get_cipher()
    if cipher:
        token = cipher.encrypt(password.encode("utf-8")).decode("utf-8")
        return {"scheme": "fernet", "value": token}

    logger.warning(
        "cryptography not available; storing password using reversible base64 encoding"
    )
    encoded = base64.urlsafe_b64encode(password.encode("utf-8")).decode("utf-8")
    return {"scheme": "base64", "value": encoded}


def decrypt_password(payload: Dict[str, str]) -> str:
    if not payload:
        return ""

    scheme = payload.get("scheme")
    value = payload.get("value", "")
    if not value:
        return ""

    if scheme == "fernet":
        cipher = _get_cipher()
        if not cipher:
            logger.error("Stored password uses fernet but cryptography is unavailable")
            return ""
        try:
            return cipher.decrypt(value.encode("utf-8")).decode("utf-8")
        except Exception:  # pragma: no cover - decryption failure path
            logger.exception("Failed to decrypt stored password")
            return ""

    if scheme == "base64":
        try:
            return base64.urlsafe_b64decode(value.encode("utf-8")).decode("utf-8")
        except Exception:  # pragma: no cover - decode failure path
            logger.exception("Failed to decode base64 password")
            return ""

    return ""


def load_saved_connection(config_path: Path) -> Dict[str, object]:
    if not config_path.exists():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:  # pragma: no cover - IO failure path
        logger.exception("Failed to load configuration file from %s", config_path)
        return {}

    password_payload = raw.get("password")
    if isinstance(password_payload, dict):
        raw["password"] = decrypt_password(password_payload)
    else:
        raw["password"] = ""

    return raw


def save_connection_config(config: Dict[str, object], config_path: Path) -> None:
    to_store = {**config}
    to_store["password"] = encrypt_password(config.get("password", ""))

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(to_store, handle, indent=2)
        logger.info("Saved Elasticsearch connection configuration to %s", config_path)
    except Exception:  # pragma: no cover - IO failure path
        logger.exception("Failed to save configuration to %s", config_path)
        raise
