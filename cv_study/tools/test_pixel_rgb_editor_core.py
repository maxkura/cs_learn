"""Tests for pixel_rgb_editor core image operations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pixel_rgb_editor.core import (
    RGBImageDocument,
    Rect,
    clamp_channel,
    clamp_rgb,
    default_export_path,
)


class RGBImageDocumentTest(unittest.TestCase):
    def make_document(self) -> RGBImageDocument:
        image = np.array(
            [
                [[0, 0, 0], [10, 20, 30], [40, 50, 60], [70, 80, 90]],
                [[1, 2, 3], [11, 22, 33], [44, 55, 66], [77, 88, 99]],
                [[5, 6, 7], [15, 26, 37], [48, 59, 70], [81, 92, 103]],
            ],
            dtype=np.uint8,
        )
        return RGBImageDocument(image)

    def test_clamps_rgb_values(self) -> None:
        self.assertEqual(clamp_channel(-5), 0)
        self.assertEqual(clamp_channel(260), 255)
        self.assertEqual(clamp_channel(12.4), 12)
        self.assertEqual(clamp_rgb((-1, 128, 999)), (0, 128, 255))

    def test_single_pixel_edit_and_undo_redo(self) -> None:
        document = self.make_document()
        self.assertTrue(document.set_pixel(1, 1, (200, 201, 202)))
        self.assertEqual(document.get_pixel(1, 1), (200, 201, 202))
        self.assertTrue(document.can_undo())

        undone = document.undo()
        self.assertEqual(undone, Rect.single_pixel(1, 1))
        self.assertEqual(document.get_pixel(1, 1), (11, 22, 33))
        self.assertTrue(document.can_redo())

        redone = document.redo()
        self.assertEqual(redone, Rect.single_pixel(1, 1))
        self.assertEqual(document.get_pixel(1, 1), (200, 201, 202))

    def test_region_set_offset_and_sample_fill(self) -> None:
        document = self.make_document()
        rect = Rect(1, 0, 2, 2)

        self.assertTrue(document.set_region(rect, (9, 8, 7)))
        np.testing.assert_array_equal(
            document.image[0:2, 1:3],
            np.array(
                [
                    [[9, 8, 7], [9, 8, 7]],
                    [[9, 8, 7], [9, 8, 7]],
                ],
                dtype=np.uint8,
            ),
        )

        self.assertTrue(document.offset_region(rect, (250, -20, 1)))
        np.testing.assert_array_equal(
            document.image[0:2, 1:3],
            np.array(
                [
                    [[255, 0, 8], [255, 0, 8]],
                    [[255, 0, 8], [255, 0, 8]],
                ],
                dtype=np.uint8,
            ),
        )

        self.assertTrue(document.fill_region_from_sample(rect, 0, 2))
        np.testing.assert_array_equal(
            document.image[0:2, 1:3],
            np.array(
                [
                    [[5, 6, 7], [5, 6, 7]],
                    [[5, 6, 7], [5, 6, 7]],
                ],
                dtype=np.uint8,
            ),
        )

    def test_copy_paste_clips_to_image_bounds(self) -> None:
        document = self.make_document()
        clipboard = document.copy_region(Rect(1, 0, 3, 2))
        pasted = document.paste_region(clipboard, 2, 2)

        self.assertEqual(pasted, Rect(2, 2, 2, 1))
        np.testing.assert_array_equal(
            document.image[2:3, 2:4],
            np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8),
        )

        document.undo()
        np.testing.assert_array_equal(
            document.image[2:3, 2:4],
            np.array([[[48, 59, 70], [81, 92, 103]]], dtype=np.uint8),
        )

    def test_paste_outside_image_returns_none(self) -> None:
        document = self.make_document()
        clipboard = document.copy_region(Rect.single_pixel(0, 0))
        self.assertIsNone(document.paste_region(clipboard, 99, 99))

    def test_clipboard_pastes_across_documents_as_snapshot(self) -> None:
        source = self.make_document()
        target = RGBImageDocument(np.zeros((3, 4, 3), dtype=np.uint8))
        clipboard = source.copy_region(Rect(1, 0, 2, 2))

        source.set_region(Rect(1, 0, 2, 2), (9, 9, 9))
        pasted = target.paste_region(clipboard, 1, 1)

        self.assertEqual(pasted, Rect(1, 1, 2, 2))
        np.testing.assert_array_equal(
            target.image[1:3, 1:3],
            np.array(
                [
                    [[10, 20, 30], [40, 50, 60]],
                    [[11, 22, 33], [44, 55, 66]],
                ],
                dtype=np.uint8,
            ),
        )
        np.testing.assert_array_equal(
            source.image[0:2, 1:3],
            np.array(
                [
                    [[9, 9, 9], [9, 9, 9]],
                    [[9, 9, 9], [9, 9, 9]],
                ],
                dtype=np.uint8,
            ),
        )

        target.undo()
        np.testing.assert_array_equal(target.image[1:3, 1:3], np.zeros((2, 2, 3), dtype=np.uint8))
        np.testing.assert_array_equal(
            source.image[0:2, 1:3],
            np.array(
                [
                    [[9, 9, 9], [9, 9, 9]],
                    [[9, 9, 9], [9, 9, 9]],
                ],
                dtype=np.uint8,
            ),
        )

    def test_default_export_path_adds_suffix_and_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "sample.jpg"
            source.touch()
            first = Path(tmp_dir) / "sample_edited.png"
            first.touch()

            self.assertEqual(default_export_path(source), Path(tmp_dir) / "sample_edited_1.png")


if __name__ == "__main__":
    unittest.main()
