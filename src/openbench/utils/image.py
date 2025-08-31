"""Image utility functions for OpenBench."""


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
