"""
gemini.py — Direct HTTP client for Google Gemini API.

Handles both text generation (streaming) and text embedding,
with retry logic and rate limiting. No Google SDK — pure requests.
"""

import json
import os
import time
from typing import Generator, List, Optional

import requests
from dotenv import load_dotenv

from config import (
    BACKOFF_SCHEDULE,
    EMBEDDING_MODEL,
    GEMINI_API_BASE,
    GENERATION_MODEL,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    embedding_rate_limiter,
    generation_rate_limiter,
)
from display import show_error, show_rate_limit, show_warning

load_dotenv()


def _get_api_key() -> str:
    """Retrieve the Gemini API key from environment.

    Returns:
        The API key string.

    Raises:
        SystemExit: If the key is not set.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key == "your_key_here":
        show_error("GEMINI_API_KEY not set. Run: python setup.py")
        raise SystemExit(1)
    return key


def embed_text(text: str) -> Optional[List[float]]:
    """Embed a text string using the Gemini gemini-embedding-001 model.

    Args:
        text: The text to embed.

    Returns:
        A list of floats (embedding vector), or None on failure.
    """
    api_key = _get_api_key()
    url = (
        f"{GEMINI_API_BASE}/models/{EMBEDDING_MODEL}:embedContent"
        f"?key={api_key}"
    )
    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {
            "parts": [{"text": text}]
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            embedding_rate_limiter.wait_if_needed()
            resp = requests.post(
                url, json=payload, timeout=REQUEST_TIMEOUT
            )
            embedding_rate_limiter.record_call()

            if resp.status_code == 200:
                data = resp.json()
                return data["embedding"]["values"]

            if resp.status_code == 429:
                wait = BACKOFF_SCHEDULE[min(attempt - 1,
                                            len(BACKOFF_SCHEDULE) - 1)]
                show_rate_limit(attempt, MAX_RETRIES, wait)
                time.sleep(wait)
                continue

            if resp.status_code == 401:
                show_error("Invalid API key (401). Run: python setup.py")
                return None

            if resp.status_code in (500, 503):
                if attempt == 1:
                    time.sleep(3)
                    continue
                show_error(f"Server error ({resp.status_code}).")
                return None

            show_error(f"Embedding failed: HTTP {resp.status_code}")
            return None

        except requests.exceptions.Timeout:
            show_error("Embedding request timed out.")
            return None
        except requests.exceptions.RequestException as exc:
            show_error(f"Embedding request failed: {exc}")
            return None

    show_error("All embedding retries exhausted.")
    return None


def embed_texts_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """Embed a batch of texts using Gemini's batchEmbedContents endpoint.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (or None for failures).
    """
    if not texts:
        return []

    api_key = _get_api_key()
    url = (
        f"{GEMINI_API_BASE}/models/{EMBEDDING_MODEL}:batchEmbedContents"
        f"?key={api_key}"
    )

    results: List[Optional[List[float]]] = [None] * len(texts)
    batch_size = 20  # fewer large batches = fewer total API requests

    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx : batch_idx + batch_size]
        payload = {
            "requests": [
                {
                    "model": f"models/{EMBEDDING_MODEL}",
                    "content": {
                        "parts": [{"text": t}]
                    }
                }
                for t in batch_texts
            ]
        }

        batch_success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                embedding_rate_limiter.wait_if_needed()
                resp = requests.post(
                    url, json=payload, timeout=60
                )
                embedding_rate_limiter.record_call()
                time.sleep(1)  # brief pause between batches

                if resp.status_code == 200:
                    data = resp.json()
                    embeddings_data = data.get("embeddings", [])
                    for i, emb_item in enumerate(embeddings_data):
                        results[batch_idx + i] = emb_item.get("values")
                    batch_success = True
                    break

                if resp.status_code == 429:
                    wait = BACKOFF_SCHEDULE[min(attempt - 1,
                                                 len(BACKOFF_SCHEDULE) - 1)]
                    show_rate_limit(attempt, MAX_RETRIES, wait)
                    time.sleep(wait)
                    continue

                if resp.status_code == 401:
                    show_error("Invalid API key (401). Run: python setup.py")
                    return results

                if resp.status_code in (500, 503):
                    if attempt == 1:
                        time.sleep(3)
                        continue
                    show_error(f"Server error ({resp.status_code}).")
                    break

                show_error(f"Batch embedding failed: HTTP {resp.status_code}")
                break

            except requests.exceptions.Timeout:
                if attempt == 1:
                    time.sleep(3)
                    continue
                show_error("Batch embedding request timed out.")
                break
            except Exception as e:
                show_error(f"Batch embedding network error: {e}")
                break

        if not batch_success:
            pass

    return results


def stream_generate(prompt: str,
                    history: Optional[List[dict]] = None
                    ) -> Generator[str, None, None]:
    """Stream a text generation response from Gemini Flash.

    Args:
        prompt: The assembled prompt string.
        history: Optional conversation history as list of
                 {"role": str, "parts": [{"text": str}]} dicts.

    Yields:
        Text chunks as they arrive from the API.

    Raises:
        GenerationError: If all retries are exhausted.
    """
    api_key = _get_api_key()
    url = (
        f"{GEMINI_API_BASE}/models/{GENERATION_MODEL}:streamGenerateContent"
        f"?alt=sse&key={api_key}"
    )

    contents = []
    if history:
        contents.extend(history)
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.3,
            "topP": 0.95,
            "maxOutputTokens": 2048,
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            generation_rate_limiter.wait_if_needed()
            resp = requests.post(
                url, json=payload, timeout=60, stream=True
            )
            generation_rate_limiter.record_call()

            if resp.status_code == 200:
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    # SSE format: lines starting with "data: "
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            return
                        try:
                            data = json.loads(data_str)
                            candidates = data.get("candidates", [])
                            if candidates:
                                content = candidates[0].get("content", {})
                                parts = content.get("parts", [])
                                for part in parts:
                                    text = part.get("text", "")
                                    if text:
                                        yield text
                        except json.JSONDecodeError:
                            continue
                return

            if resp.status_code == 429:
                wait = BACKOFF_SCHEDULE[min(attempt - 1,
                                            len(BACKOFF_SCHEDULE) - 1)]
                show_rate_limit(attempt, MAX_RETRIES, wait)
                time.sleep(wait)
                continue

            if resp.status_code == 401:
                show_error("Invalid API key (401). Run: python setup.py")
                return

            if resp.status_code in (500, 503):
                if attempt == 1:
                    time.sleep(3)
                    continue
                show_error(f"Server error ({resp.status_code}).")
                return

            show_error(f"Generation failed: HTTP {resp.status_code}")
            return

        except requests.exceptions.Timeout:
            show_error("Generation request timed out.")
            return
        except requests.exceptions.RequestException as exc:
            show_error(f"Generation request failed: {exc}")
            return

    show_error("All generation retries exhausted. Try again in a minute.")


def test_api_key(api_key: str) -> bool:
    """Make a lightweight test call to verify a Gemini API key works.

    Args:
        api_key: The API key to test.

    Returns:
        True if the key is valid and working.
    """
    url = (
        f"{GEMINI_API_BASE}/models/{GENERATION_MODEL}:generateContent"
        f"?key={api_key}"
    )
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "Say OK"}]}
        ],
        "generationConfig": {"maxOutputTokens": 5},
    }

    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return True
        if resp.status_code == 401:
            show_error("API key is invalid (401).")
        elif resp.status_code == 429:
            show_warning("Rate limited during key test — key format looks OK.")
            return True  # key is valid, just rate-limited
        else:
            show_error(f"Key test failed: HTTP {resp.status_code}")
        return False
    except requests.exceptions.RequestException as exc:
        show_error(f"Key test failed: {exc}")
        return False
