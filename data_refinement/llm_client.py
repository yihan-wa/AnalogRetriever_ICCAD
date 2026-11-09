"""
OpenRouter API client for GPT-5.4 calls.
Supports both text-only and vision (image+text) requests.
"""

import os
import json
import base64
import time
from pathlib import Path

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-5.4"
TEMPERATURE = 0.3
MAX_TOKENS = 4096


def get_api_key():
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. "
            "Run: export OPENROUTER_API_KEY='your-key'"
        )
    return key


def encode_image(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(ext, "image/jpeg")


def call_llm(
    system_prompt: str,
    user_prompt: str,
    image_path: str = None,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    max_retries: int = 3,
) -> str:
    """
    Call GPT-5.4 via OpenRouter API.
    Optionally include an image for vision tasks.
    """
    import requests

    api_key = get_api_key()

    messages = [{"role": "system", "content": system_prompt}]

    if image_path and os.path.exists(image_path):
        b64_img = encode_image(image_path)
        media_type = get_image_media_type(image_path)
        user_content = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{b64_img}"
                },
            },
        ]
    else:
        user_content = user_prompt

    messages.append({"role": "user", "content": user_content})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/analog-retriever",
        "X-Title": "AnalogRetriever Data Refinement",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"  API error {resp.status_code}: {resp.text[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
        except requests.exceptions.Timeout:
            print(f"  Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + 1)
                continue
            return None
        except (requests.exceptions.ConnectionError, requests.exceptions.SSLError) as e:
            print(f"  Connection/SSL error on attempt {attempt + 1}/{max_retries}: {type(e).__name__}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + 1)
                continue
            return None
        except Exception as e:
            print(f"  Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + 1)
                continue
            return None

    return None
