"""Pixel-level RGB image inspection and repair tool."""

from .core import ClipboardRegion, RGBImageDocument, Rect, clamp_channel, clamp_rgb

__all__ = [
    "ClipboardRegion",
    "RGBImageDocument",
    "Rect",
    "clamp_channel",
    "clamp_rgb",
]
