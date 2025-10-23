# app/security.py
import base64, os
from fastapi import HTTPException, Request
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .settings import settings

# --------- 访问控制 ----------
def require_bearer(request: Request):
    if not settings.require_auth:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    expected = (settings.api_bearer_token or "").strip()
    if not expected or token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")

# --------- 加解密 ----------
def _get_aesgcm():
    key_b64 = (settings.encryption_key_b64 or "").strip()
    if not key_b64:
        return None
    key = base64.urlsafe_b64decode(key_b64)
    return AESGCM(key)

def encrypt_bytes(plain: bytes) -> bytes:
    from secrets import token_bytes
    aes = _get_aesgcm()
    if settings.encrypt_data and aes is None:
        raise RuntimeError("ENCRYPT_DATA=true but no ENCRYPTION_KEY_B64 provided")
    if aes is None:
        return plain
    nonce = token_bytes(12)
    return nonce + aes.encrypt(nonce, plain, b"")

def decrypt_bytes(blob: bytes) -> bytes:
    aes = _get_aesgcm()
    if settings.encrypt_data and aes is None:
        raise RuntimeError("ENCRYPT_DATA=true but no ENCRYPTION_KEY_B64 provided")
    if aes is None:
        return blob
    if len(blob) < 13:
        raise ValueError("ciphertext too short")
    nonce, ct = blob[:12], blob[12:]
    return aes.decrypt(nonce, ct, b"")