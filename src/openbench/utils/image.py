"""Image utility functions for openbench."""

import base64
import io
from typing import Any

from PIL import Image


def compress_image(
    image_bytes: bytes,
    max_size_mb: float = 20.0,
    quality: int = 85,
    max_dimension: int = 2048,
) -> bytes:
    """
    Compress an image if it's too large for API requests.

    Args:
        image_bytes: Raw image bytes
        max_size_mb: Maximum allowed size in MB before compression
        quality: JPEG quality (1-100) for compression
        max_dimension: Maximum width/height in pixels

    Returns:
        Compressed image bytes (or original if small enough)
    """
    # Check if image is already small enough
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb <= max_size_mb:
        return image_bytes

    try:
        # Open image with PIL
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Convert to RGB if necessary (for JPEG compatibility)
            if img.mode in ("RGBA", "LA", "P"):
                # Create white background for transparency
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(
                    img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None
                )
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Resize if too large
            if max(img.size) > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

            # Compress to JPEG
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=quality, optimize=True)
            compressed_bytes = output.getvalue()

            # Return compressed version if it's actually smaller
            if len(compressed_bytes) < len(image_bytes):
                return compressed_bytes
            else:
                return image_bytes

    except Exception:
        # If compression fails, return original
        return image_bytes


def detect_image_mime_type(image_bytes: bytes) -> str:
    """
    Detect the MIME type of an image from its bytes.

    Uses magic bytes to detect common image formats.
    Falls back to 'image/png' if detection fails.

    Args:
        image_bytes: Raw image bytes

    Returns:
        MIME type string (e.g., 'image/png', 'image/jpeg', 'image/webp')
    """
    try:
        # Use magic bytes to detect image format
        return _detect_from_magic_bytes(image_bytes)

    except Exception:
        # Fallback to PNG if detection fails
        return "image/png"


def _detect_from_magic_bytes(image_bytes: bytes) -> str:
    """
    Detect image format from magic bytes.

    Args:
        image_bytes: Raw image bytes

    Returns:
        MIME type string
    """
    if len(image_bytes) < 4:
        return "image/png"

    # Check for common image format signatures
    signatures = [
        (b"\xff\xd8\xff", "image/jpeg"),  # JPEG
        (b"\x89PNG\r\n\x1a\n", "image/png"),  # PNG
        (b"GIF87a", "image/gif"),  # GIF87a
        (b"GIF89a", "image/gif"),  # GIF89a
        (b"BM", "image/bmp"),  # BMP
        (b"RIFF", "image/webp"),  # WebP (RIFF header)
        (b"II*\x00", "image/tiff"),  # TIFF little-endian
        (b"MM\x00*", "image/tiff"),  # TIFF big-endian
        (b"\x00\x00\x01\x00", "image/ico"),  # ICO
        (b"\x00\x00\x02\x00", "image/ico"),  # ICO
    ]

    for signature, mime_type in signatures:
        if image_bytes.startswith(signature):
            return mime_type

    # Default fallback
    return "image/png"


def pil_image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """
    Convert a PIL Image object to bytes.

    Args:
        image: PIL Image object
        format: Image format to save as (e.g., "PNG", "JPEG")

    Returns:
        Image bytes

    Examples:
        >>> from PIL import Image
        >>> img = Image.new('RGB', (100, 100))
        >>> img_bytes = pil_image_to_bytes(img)
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()


def extract_image_bytes(image_data: Any) -> bytes:
    """
    Extract bytes from various image data formats.

    Handles multiple input formats commonly encountered when loading images
    from HuggingFace datasets:
    - HuggingFace dict format: {"bytes": b"...", "path": "..."}
    - Raw bytes: b"..."
    - PIL Image objects: Image.Image

    Args:
        image_data: Image data in various formats

    Returns:
        Raw image bytes

    Raises:
        ValueError: If image_data format is not supported

    Examples:
        >>> # HuggingFace format
        >>> image_bytes = extract_image_bytes({"bytes": b"...", "path": "..."})
        >>>
        >>> # Raw bytes
        >>> image_bytes = extract_image_bytes(b"...")
        >>>
        >>> # PIL Image
        >>> from PIL import Image
        >>> img = Image.new('RGB', (100, 100))
        >>> image_bytes = extract_image_bytes(img)
    """
    # Handle HuggingFace dataset dict format
    if isinstance(image_data, dict) and "bytes" in image_data:
        return image_data["bytes"]

    # Handle raw bytes
    if isinstance(image_data, bytes):
        return image_data

    # Handle PIL Image
    if isinstance(image_data, Image.Image):
        return pil_image_to_bytes(image_data, format="PNG")

    # Unsupported format
    raise ValueError(
        f"Unsupported image data format: {type(image_data)}. "
        "Expected dict with 'bytes' key, raw bytes, or PIL Image."
    )


def image_bytes_to_data_uri(image_bytes: bytes) -> str:
    """
    Convert image bytes to a base64-encoded data URI.

    Automatically detects the image MIME type and creates a properly
    formatted data URI suitable for use in HTML or API requests.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Data URI string (e.g., "data:image/png;base64,iVBORw0KG...")

    Examples:
        >>> with open("image.png", "rb") as f:
        ...     image_bytes = f.read()
        >>> data_uri = image_bytes_to_data_uri(image_bytes)
        >>> data_uri.startswith("data:image/png;base64,")
        True
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = detect_image_mime_type(image_bytes)
    return f"data:{mime_type};base64,{base64_image}"
