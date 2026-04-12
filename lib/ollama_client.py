"""Ollama vision API client for image description."""

import requests

from lib.image_utils import image_to_base64, validate_image

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2-vision"
TIMEOUT = 120  # seconds

DESCRIBE_PROMPT = (
    "Describe the main character(s) in this image in detail: "
    "hair color and style, eye color, skin tone, clothing and outfit, "
    "any distinctive accessories or features, and overall color palette. "
    "Be specific."
)


def check_ollama_available() -> tuple[bool, str]:
    """Check if Ollama is running and the vision model is available."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        # Check for the model name with or without tag
        has_model = any(MODEL in m for m in models)
        if not has_model:
            return False, (
                f"Model '{MODEL}' not found. Available models: {', '.join(models)}. "
                f"Run: ollama pull {MODEL}"
            )
        return True, ""
    except requests.ConnectionError:
        return False, "Cannot connect to Ollama at localhost:11434. Is it running?"
    except Exception as e:
        return False, f"Error checking Ollama: {e}"


def describe_image(image_path: str) -> tuple[str, str]:
    """
    Send an image to Ollama for description.
    Returns (description, error_message). On success error_message is empty.
    """
    ok, err = validate_image(image_path)
    if not ok:
        return "", f"Corrupt image: {err}"

    try:
        b64 = image_to_base64(image_path)
    except Exception as e:
        return "", f"Cannot read image: {e}"

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": DESCRIBE_PROMPT,
                "images": [b64],
            }
        ],
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        if not content:
            return "", "Empty response from Ollama"
        return content.strip(), ""
    except requests.Timeout:
        return "", f"Ollama timed out after {TIMEOUT}s"
    except requests.ConnectionError:
        return "", "Lost connection to Ollama"
    except requests.HTTPError as e:
        return "", f"Ollama HTTP error: {e}"
    except Exception as e:
        return "", f"Unexpected error: {e}"
