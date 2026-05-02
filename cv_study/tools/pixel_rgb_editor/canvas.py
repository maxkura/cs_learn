"""Qt canvas for zooming, panning, selecting, and previewing RGB pixels."""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget

from .core import Rect, RGBImageDocument


class ImageCanvas(QWidget):
    hoverChanged = Signal(object)
    selectionChanged = Signal(object)
    pasteTargetChanged = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.document: RGBImageDocument | None = None
        self.zoom = 1.0
        self.offset = QPointF(20.0, 20.0)
        self.selection: Rect | None = None
        self.hover_pixel: tuple[int, int] | None = None
        self._drag_start_pixel: tuple[int, int] | None = None
        self._is_selecting = False
        self._is_panning = False
        self._last_mouse_pos = QPoint()
        self._paste_size: tuple[int, int] | None = None
        self.paste_origin: tuple[int, int] | None = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(480, 360)

    def set_document(self, document: RGBImageDocument | None) -> None:
        self.document = document
        self.selection = None
        self.hover_pixel = None
        self._paste_size = None
        self.paste_origin = None
        if document is not None:
            self.fit_to_view()
        self.selectionChanged.emit(self.selection)
        self.hoverChanged.emit(self.hover_pixel)
        self.update()

    def set_selection(self, rect: Rect | None) -> None:
        self.selection = rect
        self.selectionChanged.emit(self.selection)
        self.update()

    def begin_paste_preview(self, width: int, height: int) -> None:
        self._paste_size = (width, height)
        self.paste_origin = self.hover_pixel
        self.pasteTargetChanged.emit(self.paste_origin)
        self.update()

    def clear_paste_preview(self) -> None:
        self._paste_size = None
        self.paste_origin = None
        self.pasteTargetChanged.emit(None)
        self.update()

    def fit_to_view(self) -> None:
        if self.document is None or self.document.width == 0 or self.document.height == 0:
            return
        margin = 32
        available_width = max(1, self.width() - margin * 2)
        available_height = max(1, self.height() - margin * 2)
        self.zoom = min(
            available_width / self.document.width,
            available_height / self.document.height,
        )
        self.zoom = max(0.05, min(32.0, self.zoom))
        image_width = self.document.width * self.zoom
        image_height = self.document.height * self.zoom
        self.offset = QPointF(
            (self.width() - image_width) / 2.0,
            (self.height() - image_height) / 2.0,
        )
        self.update()

    def image_to_screen_rect(self, rect: Rect) -> QRectF:
        return QRectF(
            self.offset.x() + rect.x * self.zoom,
            self.offset.y() + rect.y * self.zoom,
            rect.width * self.zoom,
            rect.height * self.zoom,
        )

    def screen_to_pixel(self, pos: QPointF) -> tuple[int, int] | None:
        if self.document is None:
            return None
        x = math.floor((pos.x() - self.offset.x()) / self.zoom)
        y = math.floor((pos.y() - self.offset.y()) / self.zoom)
        if 0 <= x < self.document.width and 0 <= y < self.document.height:
            return int(x), int(y)
        return None

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(32, 34, 38))

        if self.document is None:
            painter.setPen(QColor(200, 205, 212))
            painter.drawText(self.rect(), Qt.AlignCenter, "Open a PNG or JPG image")
            return

        image = self.document.image
        qimage = QImage(
            image.data,
            self.document.width,
            self.document.height,
            int(image.strides[0]),
            QImage.Format_RGB888,
        )
        target = QRectF(
            self.offset.x(),
            self.offset.y(),
            self.document.width * self.zoom,
            self.document.height * self.zoom,
        )
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawImage(target, qimage)

        visible = self._visible_pixel_rect()
        if visible is not None and self.zoom >= 8:
            self._draw_grid(painter, visible)
        if visible is not None and self.zoom >= 34 and visible.area <= 5000:
            self._draw_rgb_text(painter, visible)
        if self.selection is not None:
            self._draw_selection(painter, self.selection, QColor(255, 204, 77))
        if self._paste_size is not None and self.paste_origin is not None:
            width, height = self._paste_size
            preview = Rect(self.paste_origin[0], self.paste_origin[1], width, height)
            clipped = preview.clipped_to(self.document.width, self.document.height)
            self._draw_selection(painter, preview, QColor(82, 209, 127))
            if clipped is not None:
                painter.fillRect(self.image_to_screen_rect(clipped), QColor(82, 209, 127, 45))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if self.document is not None and self.zoom <= 1.0:
            self.fit_to_view()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt override
        if self.document is None:
            return
        old_zoom = self.zoom
        steps = event.angleDelta().y() / 120.0
        factor = 1.2 ** steps
        new_zoom = max(0.05, min(80.0, old_zoom * factor))
        if new_zoom == old_zoom:
            return

        cursor = event.position()
        image_x = (cursor.x() - self.offset.x()) / old_zoom
        image_y = (cursor.y() - self.offset.y()) / old_zoom
        self.zoom = new_zoom
        self.offset = QPointF(cursor.x() - image_x * new_zoom, cursor.y() - image_y * new_zoom)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        self.setFocus()
        self._last_mouse_pos = event.pos()
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._is_panning = True
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() != Qt.LeftButton:
            return
        pixel = self.screen_to_pixel(event.position())
        if pixel is None:
            return
        if self._paste_size is not None:
            self.paste_origin = pixel
            self.pasteTargetChanged.emit(self.paste_origin)
            self.update()
            return

        self._drag_start_pixel = pixel
        self._is_selecting = True
        self.set_selection(Rect.single_pixel(*pixel))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        pixel = self.screen_to_pixel(event.position())
        if pixel != self.hover_pixel:
            self.hover_pixel = pixel
            self.hoverChanged.emit(self.hover_pixel)

        if self._is_panning:
            delta = event.pos() - self._last_mouse_pos
            self.offset += QPointF(delta)
            self._last_mouse_pos = event.pos()
            self.update()
            return

        if self._paste_size is not None and pixel is not None:
            self.paste_origin = pixel
            self.pasteTargetChanged.emit(self.paste_origin)
            self.update()
            return

        if self._is_selecting and self._drag_start_pixel is not None and pixel is not None:
            start_x, start_y = self._drag_start_pixel
            self.set_selection(Rect.from_points(start_x, start_y, pixel[0], pixel[1]))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._is_panning = False
            self.unsetCursor()
        if event.button() == Qt.LeftButton:
            self._is_selecting = False
            self._drag_start_pixel = None

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.hover_pixel = None
        self.hoverChanged.emit(self.hover_pixel)

    def _visible_pixel_rect(self) -> Rect | None:
        if self.document is None:
            return None
        left = max(0, int(math.floor((-self.offset.x()) / self.zoom)))
        top = max(0, int(math.floor((-self.offset.y()) / self.zoom)))
        right = min(self.document.width, int(math.ceil((self.width() - self.offset.x()) / self.zoom)))
        bottom = min(self.document.height, int(math.ceil((self.height() - self.offset.y()) / self.zoom)))
        if right <= left or bottom <= top:
            return None
        return Rect(left, top, right - left, bottom - top)

    def _draw_grid(self, painter: QPainter, visible: Rect) -> None:
        painter.setPen(QPen(QColor(255, 255, 255, 70), 1))
        for x in range(visible.x, visible.right + 1):
            sx = self.offset.x() + x * self.zoom
            painter.drawLine(QPointF(sx, self.offset.y() + visible.y * self.zoom), QPointF(sx, self.offset.y() + visible.bottom * self.zoom))
        for y in range(visible.y, visible.bottom + 1):
            sy = self.offset.y() + y * self.zoom
            painter.drawLine(QPointF(self.offset.x() + visible.x * self.zoom, sy), QPointF(self.offset.x() + visible.right * self.zoom, sy))

    def _draw_rgb_text(self, painter: QPainter, visible: Rect) -> None:
        assert self.document is not None
        font = painter.font()
        font.setPointSize(max(6, min(10, int(self.zoom / 5))))
        painter.setFont(font)
        for y in range(visible.y, visible.bottom):
            for x in range(visible.x, visible.right):
                r, g, b = self.document.get_pixel(x, y)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                painter.setPen(QColor(20, 22, 26) if luminance > 150 else QColor(245, 247, 250))
                painter.drawText(
                    self.image_to_screen_rect(Rect.single_pixel(x, y)).adjusted(1, 1, -1, -1),
                    Qt.AlignCenter,
                    f"{r},{g},{b}",
                )

    def _draw_selection(self, painter: QPainter, rect: Rect, color: QColor) -> None:
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self.image_to_screen_rect(rect))
