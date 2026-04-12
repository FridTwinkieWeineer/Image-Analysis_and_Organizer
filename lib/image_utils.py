"""Image discovery, encoding, and thumbnail utilities."""

import base64
import os
from io import BytesIO
from pathlib import Path

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def find_images(root: str) -> list[str]:
    """Recursively find all supported image files under root."""
    images = []
    root_path = Path(root)
    if not root_path.exists():
        return images
    for dirpath, _, filenames in os.walk(root_path):
        for fname in sorted(filenames):
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(str(Path(dirpath) / fname))
    return images


def image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def make_thumbnail(image_path: str, size: tuple[int, int] = (200, 200)) -> Image.Image:
    """Create a PIL thumbnail for display."""
    img = Image.open(image_path)
    img.thumbnail(size, Image.LANCZOS)
    return img


def validate_image(image_path: str) -> tuple[bool, str]:
    """Check if an image file can be opened. Returns (ok, error_message)."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True, ""
    except Exception as e:
        return False, str(e)
