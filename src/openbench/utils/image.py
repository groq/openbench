"""Image utility functions for OpenBench."""

import io
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
