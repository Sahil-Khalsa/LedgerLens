"""
API key authentication.
Keys are stored as SHA-256 hashes only — never plaintext.

Generate a new key:
  uv run python -m api.auth --generate
"""

import hashlib
import secrets

from fastapi import Header, HTTPException

from config import settings


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def verify_api_key(x_api_key: str = Header(...)) -> str:
    hashed = hash_key(x_api_key)
    if not settings.valid_api_key_hashes or hashed not in settings.valid_api_key_hashes:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    if args.generate:
        key = secrets.token_urlsafe(32)
        hashed = hash_key(key)
        print(f"Key:  {key}")
        print(f"Hash: {hashed}")
        print("\nAdd the hash to API_KEY_HASHES in your .env file.")


if __name__ == "__main__":
    _cli()
