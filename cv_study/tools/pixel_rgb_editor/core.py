"""Core image buffer operations for the pixel RGB editor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def clamp_channel(value: int | float) -> int:
    """Return an integer channel value clipped to the 0-255 RGB range."""
    return max(0, min(255, int(round(value))))


def clamp_rgb(values: Iterable[int | float]) -> tuple[int, int, int]:
    rgb = tuple(clamp_channel(value) for value in values)
    if len(rgb) != 3:
        raise ValueError("RGB values must contain exactly three channels.")
    return rgb


@dataclass(frozen=True)
class Rect:
    """Integer image rectangle using x/y plus width/height."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError("Rect width and height must be non-negative.")

    @classmethod
    def from_points(cls, x1: int, y1: int, x2: int, y2: int) -> "Rect":
        left = min(x1, x2)
        top = min(y1, y2)
        return cls(left, top, abs(x2 - x1) + 1, abs(y2 - y1) + 1)

    @classmethod
    def single_pixel(cls, x: int, y: int) -> "Rect":
        return cls(x, y, 1, 1)

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0

    def clipped_to(self, width: int, height: int) -> Optional["Rect"]:
        left = max(0, self.x)
        top = max(0, self.y)
        right = min(width, self.right)
        bottom = min(height, self.bottom)
        if right <= left or bottom <= top:
            return None
        return Rect(left, top, right - left, bottom - top)

    def as_slices(self) -> tuple[slice, slice]:
        return slice(self.y, self.bottom), slice(self.x, self.right)

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.right and self.y <= y < self.bottom


@dataclass(frozen=True)
class ClipboardRegion:
    """Copied RGB pixels ready for previewed paste."""

    data: np.ndarray

    def __post_init__(self) -> None:
        array = np.asarray(self.data)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Clipboard data must be an HxWx3 RGB array.")
        if array.dtype != np.uint8:
            raise ValueError("Clipboard data must use uint8 RGB values.")
        object.__setattr__(self, "data", np.ascontiguousarray(array.copy()))

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        return int(self.data.shape[0])


@dataclass
class ImagePatch:
    rect: Rect
    before: np.ndarray
    after: np.ndarray

    def apply(self, image: np.ndarray) -> None:
        y_slice, x_slice = self.rect.as_slices()
        image[y_slice, x_slice] = self.after

    def revert(self, image: np.ndarray) -> None:
        y_slice, x_slice = self.rect.as_slices()
        image[y_slice, x_slice] = self.before


class RGBImageDocument:
    """Editable 8-bit RGB image plus undo/redo history."""

    def __init__(self, image: np.ndarray, source_path: Path | None = None) -> None:
        self.image = self._normalize_rgb_array(image)
        self.source_path = Path(source_path) if source_path is not None else None
        self.undo_stack: list[ImagePatch] = []
        self.redo_stack: list[ImagePatch] = []
        self.dirty = False

    @classmethod
    def open(cls, path: str | Path) -> "RGBImageDocument":
        image_path = Path(path)
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("Only PNG and JPG images are supported.")
        with Image.open(image_path) as pil_image:
            rgb_image = pil_image.convert("RGB")
            array = np.asarray(rgb_image, dtype=np.uint8).copy()
        return cls(array, image_path)

    @staticmethod
    def _normalize_rgb_array(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Image must be an HxWx3 RGB array.")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array.copy())

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def get_pixel(self, x: int, y: int) -> tuple[int, int, int]:
        self._require_point(x, y)
        r, g, b = self.image[y, x]
        return int(r), int(g), int(b)

    def set_pixel(self, x: int, y: int, rgb: Iterable[int | float]) -> bool:
        self._require_point(x, y)
        return self.set_region(Rect.single_pixel(x, y), clamp_rgb(rgb))

    def set_region(self, rect: Rect, rgb: Iterable[int | float]) -> bool:
        clipped = self._clip_rect(rect)
        if clipped is None:
            return False
        y_slice, x_slice = clipped.as_slices()
        before = self.image[y_slice, x_slice].copy()
        after = np.empty_like(before)
        after[...] = clamp_rgb(rgb)
        return self._commit_patch(clipped, before, after)

    def offset_region(self, rect: Rect, offsets: Iterable[int | float]) -> bool:
        clipped = self._clip_rect(rect)
        if clipped is None:
            return False
        offset = np.array(clamp_rgb_channel_offsets(offsets), dtype=np.int16)
        y_slice, x_slice = clipped.as_slices()
        before = self.image[y_slice, x_slice].copy()
        after = np.clip(before.astype(np.int16) + offset, 0, 255).astype(np.uint8)
        return self._commit_patch(clipped, before, after)

    def fill_region_from_sample(self, rect: Rect, sample_x: int, sample_y: int) -> bool:
        return self.set_region(rect, self.get_pixel(sample_x, sample_y))

    def copy_region(self, rect: Rect) -> ClipboardRegion:
        clipped = self._clip_rect(rect)
        if clipped is None:
            raise ValueError("Cannot copy an empty or out-of-bounds region.")
        y_slice, x_slice = clipped.as_slices()
        return ClipboardRegion(self.image[y_slice, x_slice].copy())

    def paste_region(
        self, clipboard: ClipboardRegion, target_x: int, target_y: int
    ) -> Optional[Rect]:
        dest_left = max(0, int(target_x))
        dest_top = max(0, int(target_y))
        dest_right = min(self.width, int(target_x) + clipboard.width)
        dest_bottom = min(self.height, int(target_y) + clipboard.height)
        if dest_right <= dest_left or dest_bottom <= dest_top:
            return None

        source_left = dest_left - int(target_x)
        source_top = dest_top - int(target_y)
        source_right = source_left + (dest_right - dest_left)
        source_bottom = source_top + (dest_bottom - dest_top)

        rect = Rect(dest_left, dest_top, dest_right - dest_left, dest_bottom - dest_top)
        y_slice, x_slice = rect.as_slices()
        before = self.image[y_slice, x_slice].copy()
        after = clipboard.data[source_top:source_bottom, source_left:source_right].copy()
        self._commit_patch(rect, before, after)
        return rect

    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def undo(self) -> Optional[Rect]:
        if not self.undo_stack:
            return None
        patch = self.undo_stack.pop()
        patch.revert(self.image)
        self.redo_stack.append(patch)
        self.dirty = True
        return patch.rect

    def redo(self) -> Optional[Rect]:
        if not self.redo_stack:
            return None
        patch = self.redo_stack.pop()
        patch.apply(self.image)
        self.undo_stack.append(patch)
        self.dirty = True
        return patch.rect

    def default_export_path(self) -> Path:
        return default_export_path(self.source_path)

    def export_png(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.image).save(output_path, format="PNG")
        self.dirty = False
        return output_path

    def _clip_rect(self, rect: Rect) -> Optional[Rect]:
        return rect.clipped_to(self.width, self.height)

    def _require_point(self, x: int, y: int) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"Pixel coordinate out of bounds: ({x}, {y})")

    def _commit_patch(self, rect: Rect, before: np.ndarray, after: np.ndarray) -> bool:
        if before.shape != after.shape:
            raise ValueError("Patch before/after arrays must have matching shapes.")
        if np.array_equal(before, after):
            return False
        patch = ImagePatch(rect, before.copy(), after.copy())
        patch.apply(self.image)
        self.undo_stack.append(patch)
        self.redo_stack.clear()
        self.dirty = True
        return True


def clamp_rgb_channel_offsets(values: Iterable[int | float]) -> tuple[int, int, int]:
    offsets = tuple(max(-255, min(255, int(round(value)))) for value in values)
    if len(offsets) != 3:
        raise ValueError("RGB offsets must contain exactly three channels.")
    return offsets


def default_export_path(source_path: str | Path | None) -> Path:
    if source_path is None:
        base = Path.cwd() / "untitled_edited.png"
    else:
        source = Path(source_path)
        base = source.with_name(f"{source.stem}_edited.png")

    if not base.exists():
        return base

    for index in range(1, 10000):
        candidate = base.with_name(f"{base.stem}_{index}{base.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available export path near {base}")
