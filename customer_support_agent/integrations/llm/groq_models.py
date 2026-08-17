"""
Dynamic Groq model discovery.

Groq periodically deprecates and retires model IDs (see
https://console.groq.com/docs/deprecations). Rather than hardcoding a model
name that can silently start failing on its shutdown date, this module asks
Groq's own API which chat-capable models are currently live and picks the
best available one. A hardcoded model ID is used only as a last resort, if
the live API call itself fails (no network, bad key, Groq outage, etc.).
"""

from __future__ import annotations

import time

import requests

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

# Model families that aren't general chat/text models — excluded from consideration.
_EXCLUDE_KEYWORDS = ("whisper", "tts", "guard", "moderation", "embed", "safeguard")

# Preferred model IDs in priority order. If present in the live model list,
# the first match is used. This keeps the chosen model stable and predictable
# instead of picking an arbitrary alphabetical result, while still adapting
# automatically the moment Groq retires one of these.
_PREFERRED_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "qwen/qwen3.6-27b",
    "llama-3.1-8b-instant",
]

# Used only if the live API call fails outright and nothing is cached.
_HARDCODED_FALLBACK = "openai/gpt-oss-120b"

_CACHE_TTL_SECONDS = 3600
_cache: dict = {"models": None, "fetched_at": 0.0}


def _fetch_live_chat_models(api_key: str) -> list[str]:
    response = requests.get(
        GROQ_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=5,
    )
    response.raise_for_status()
    models = response.json().get("data", [])
    return sorted(
        m["id"] for m in models
        if not any(kw in m["id"].lower() for kw in _EXCLUDE_KEYWORDS)
    )


def get_available_groq_models(api_key: str) -> list[str]:
    """Returns the live list of chat-capable Groq models, cached for an hour."""
    now = time.time()
    if _cache["models"] is not None and (now - _cache["fetched_at"]) < _CACHE_TTL_SECONDS:
        return _cache["models"]

    try:
        models = _fetch_live_chat_models(api_key)
        if models:
            _cache["models"] = models
            _cache["fetched_at"] = now
            return models
    except requests.RequestException:
        pass

    return _cache["models"] or [_HARDCODED_FALLBACK]


def get_default_groq_model(api_key: str) -> str:
    """
    Picks the best currently-available Groq model: the first entry from
    _PREFERRED_MODELS that Groq is currently hosting, falling back to
    whatever chat model Groq returns first if none of the preferred models
    are live, and to a hardcoded model ID if the API call fails entirely.
    """
    if not api_key:
        return _HARDCODED_FALLBACK

    available = get_available_groq_models(api_key)

    for preferred in _PREFERRED_MODELS:
        if preferred in available:
            return preferred

    return available[0] if available else _HARDCODED_FALLBACK