"""PySide6 main window for the pixel RGB editor."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .canvas import ImageCanvas
from .core import ClipboardRegion, RGBImageDocument, Rect


class MainWindow(QMainWindow):
    def __init__(self, initial_image: Path | None = None) -> None:
        super().__init__()
        self.document: RGBImageDocument | None = None
        self.clipboard: ClipboardRegion | None = None
        self.sampled_rgb: tuple[int, int, int] | None = None
        self.current_hover: tuple[int, int] | None = None
        self.current_selection: Rect | None = None

        self.canvas = ImageCanvas()
        self._build_panel()
        self._build_window()
        self._build_actions()
        self._connect_signals()

        self.setWindowTitle("Pixel RGB Editor")
        self.resize(1180, 760)
        if initial_image is not None:
            self.load_image(initial_image)
        else:
            self._refresh_panel()

    def _build_window(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Ready")

    def _build_panel(self) -> None:
        self.panel = QWidget()
        self.panel.setMinimumWidth(300)
        self.panel.setMaximumWidth(380)
        layout = QVBoxLayout(self.panel)

        self.path_label = QLabel("No image loaded")
        self.path_label.setWordWrap(True)
        self.size_label = QLabel("-")
        self.hover_label = QLabel("Hover: -")
        self.selection_label = QLabel("Selection: -")
        self.pixel_label = QLabel("Selected pixel: -")
        for label in (
            self.path_label,
            self.size_label,
            self.hover_label,
            self.selection_label,
            self.pixel_label,
        ):
            layout.addWidget(label)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        layout.addWidget(line)

        self.pixel_group = QGroupBox("Pixel Edit")
        pixel_layout = QFormLayout(self.pixel_group)
        self.pixel_r = self._channel_spinbox()
        self.pixel_g = self._channel_spinbox()
        self.pixel_b = self._channel_spinbox()
        self.apply_pixel_button = QPushButton("Apply Pixel RGB")
        pixel_layout.addRow("R", self.pixel_r)
        pixel_layout.addRow("G", self.pixel_g)
        pixel_layout.addRow("B", self.pixel_b)
        pixel_layout.addRow(self.apply_pixel_button)
        layout.addWidget(self.pixel_group)

        self.region_group = QGroupBox("Region Edit")
        region_layout = QVBoxLayout(self.region_group)
        fixed_form = QFormLayout()
        self.fixed_r = self._channel_spinbox()
        self.fixed_g = self._channel_spinbox()
        self.fixed_b = self._channel_spinbox()
        fixed_form.addRow("Set R", self.fixed_r)
        fixed_form.addRow("Set G", self.fixed_g)
        fixed_form.addRow("Set B", self.fixed_b)
        self.apply_fixed_button = QPushButton("Set Selection RGB")
        region_layout.addLayout(fixed_form)
        region_layout.addWidget(self.apply_fixed_button)

        offset_form = QFormLayout()
        self.offset_r = self._offset_spinbox()
        self.offset_g = self._offset_spinbox()
        self.offset_b = self._offset_spinbox()
        offset_form.addRow("Offset R", self.offset_r)
        offset_form.addRow("Offset G", self.offset_g)
        offset_form.addRow("Offset B", self.offset_b)
        self.apply_offset_button = QPushButton("Apply RGB Offset")
        region_layout.addLayout(offset_form)
        region_layout.addWidget(self.apply_offset_button)

        self.sample_label = QLabel("Sample: -")
        self.sample_button = QPushButton("Sample Hover Pixel")
        self.fill_sample_button = QPushButton("Fill Selection With Sample")
        region_layout.addWidget(self.sample_label)
        region_layout.addWidget(self.sample_button)
        region_layout.addWidget(self.fill_sample_button)
        layout.addWidget(self.region_group)

        paste_group = QGroupBox("Copy / Paste")
        paste_layout = QVBoxLayout(paste_group)
        self.copy_button = QPushButton("Copy Selection")
        self.paste_button = QPushButton("Paste Preview")
        self.confirm_paste_button = QPushButton("Confirm Paste")
        self.cancel_paste_button = QPushButton("Cancel Paste")
        paste_layout.addWidget(self.copy_button)
        paste_layout.addWidget(self.paste_button)
        paste_layout.addWidget(self.confirm_paste_button)
        paste_layout.addWidget(self.cancel_paste_button)
        layout.addWidget(paste_group)

        layout.addStretch(1)

    def _build_actions(self) -> None:
        open_action = QAction("Open...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_dialog)

        export_action = QAction("Export Copy...", self)
        export_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        export_action.triggered.connect(self.export_dialog)

        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo)
        self.undo_action = undo_action

        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(self.redo)
        self.redo_action = redo_action

        copy_action = QAction("Copy Selection", self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(self.copy_selection)

        paste_action = QAction("Paste Preview", self)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.triggered.connect(self.start_paste_preview)

        confirm_action = QAction("Confirm Paste", self)
        confirm_action.setShortcut(QKeySequence(Qt.Key_Return))
        confirm_action.triggered.connect(self.confirm_paste)

        cancel_action = QAction("Cancel Paste", self)
        cancel_action.setShortcut(QKeySequence(Qt.Key_Escape))
        cancel_action.triggered.connect(self.cancel_paste)

        toolbar = self.addToolBar("Main")
        for action in (
            open_action,
            export_action,
            undo_action,
            redo_action,
            copy_action,
            paste_action,
            confirm_action,
            cancel_action,
        ):
            self.addAction(action)
            toolbar.addAction(action)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_action)
        file_menu.addAction(export_action)

        edit_menu = self.menuBar().addMenu("Edit")
        for action in (
            undo_action,
            redo_action,
            copy_action,
            paste_action,
            confirm_action,
            cancel_action,
        ):
            edit_menu.addAction(action)

    def _connect_signals(self) -> None:
        self.canvas.hoverChanged.connect(self._on_hover_changed)
        self.canvas.selectionChanged.connect(self._on_selection_changed)
        self.canvas.pasteTargetChanged.connect(self._on_paste_target_changed)
        self.apply_pixel_button.clicked.connect(self.apply_pixel_edit)
        self.apply_fixed_button.clicked.connect(self.apply_fixed_region)
        self.apply_offset_button.clicked.connect(self.apply_offset_region)
        self.sample_button.clicked.connect(self.sample_hover_pixel)
        self.fill_sample_button.clicked.connect(self.fill_selection_with_sample)
        self.copy_button.clicked.connect(self.copy_selection)
        self.paste_button.clicked.connect(self.start_paste_preview)
        self.confirm_paste_button.clicked.connect(self.confirm_paste)
        self.cancel_paste_button.clicked.connect(self.cancel_paste)

    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg)",
        )
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        try:
            self.document = RGBImageDocument.open(path)
        except Exception as exc:  # pragma: no cover - UI guard
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self.canvas.set_document(self.document)
        self.clipboard = None
        self.sampled_rgb = None
        self.setWindowTitle(f"Pixel RGB Editor - {path.name}")
        self.statusBar().showMessage(f"Loaded {path}")
        self._refresh_panel()

    def export_dialog(self) -> None:
        if self.document is None:
            self.statusBar().showMessage("No image loaded")
            return
        default_path = self.document.default_export_path()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Edited Copy",
            str(default_path),
            "PNG image (*.png)",
        )
        if not path:
            return
        output = Path(path)
        if output.exists():
            answer = QMessageBox.question(
                self,
                "Overwrite existing file?",
                f"{output} already exists. Overwrite it?",
            )
            if answer != QMessageBox.Yes:
                return
        try:
            saved = self.document.export_png(output)
        except Exception as exc:  # pragma: no cover - UI guard
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported {saved}")
        self._refresh_panel()

    def apply_pixel_edit(self) -> None:
        if self.document is None or self.current_selection is None:
            return
        if self.current_selection.area != 1:
            self.statusBar().showMessage("Select exactly one pixel to edit it")
            return
        changed = self.document.set_pixel(
            self.current_selection.x,
            self.current_selection.y,
            (self.pixel_r.value(), self.pixel_g.value(), self.pixel_b.value()),
        )
        self._after_edit("Pixel updated" if changed else "Pixel already had that RGB")

    def apply_fixed_region(self) -> None:
        if self.document is None or self.current_selection is None:
            return
        changed = self.document.set_region(
            self.current_selection,
            (self.fixed_r.value(), self.fixed_g.value(), self.fixed_b.value()),
        )
        self._after_edit("Selection RGB updated" if changed else "Selection already matched")

    def apply_offset_region(self) -> None:
        if self.document is None or self.current_selection is None:
            return
        changed = self.document.offset_region(
            self.current_selection,
            (self.offset_r.value(), self.offset_g.value(), self.offset_b.value()),
        )
        self._after_edit("Selection RGB offset applied" if changed else "Offset changed nothing")

    def sample_hover_pixel(self) -> None:
        if self.document is None or self.current_hover is None:
            self.statusBar().showMessage("Hover over a pixel to sample it")
            return
        self.sampled_rgb = self.document.get_pixel(*self.current_hover)
        self._refresh_panel()
        self.statusBar().showMessage(f"Sampled RGB {self.sampled_rgb}")

    def fill_selection_with_sample(self) -> None:
        if self.document is None or self.current_selection is None or self.sampled_rgb is None:
            self.statusBar().showMessage("Need a selection and sampled RGB")
            return
        changed = self.document.set_region(self.current_selection, self.sampled_rgb)
        self._after_edit("Selection filled with sampled RGB" if changed else "Selection already matched sample")

    def copy_selection(self) -> None:
        if self.document is None or self.current_selection is None:
            self.statusBar().showMessage("No selection to copy")
            return
        self.clipboard = self.document.copy_region(self.current_selection)
        self.statusBar().showMessage(f"Copied {self.clipboard.width}x{self.clipboard.height} pixels")
        self._refresh_panel()

    def start_paste_preview(self) -> None:
        if self.clipboard is None:
            self.statusBar().showMessage("Copy a selection before pasting")
            return
        self.canvas.begin_paste_preview(self.clipboard.width, self.clipboard.height)
        self.statusBar().showMessage("Move preview to target, then press Enter to confirm or Esc to cancel")

    def confirm_paste(self) -> None:
        if self.document is None or self.clipboard is None or self.canvas.paste_origin is None:
            return
        rect = self.document.paste_region(self.clipboard, *self.canvas.paste_origin)
        self.canvas.clear_paste_preview()
        if rect is not None:
            self.canvas.set_selection(rect)
            self._after_edit(f"Pasted into {rect.width}x{rect.height} region")
        else:
            self.statusBar().showMessage("Paste target is outside the image")

    def cancel_paste(self) -> None:
        self.canvas.clear_paste_preview()
        self.statusBar().showMessage("Paste canceled")

    def undo(self) -> None:
        if self.document is None:
            return
        rect = self.document.undo()
        if rect is not None:
            self.canvas.set_selection(rect)
            self._after_edit("Undo")

    def redo(self) -> None:
        if self.document is None:
            return
        rect = self.document.redo()
        if rect is not None:
            self.canvas.set_selection(rect)
            self._after_edit("Redo")

    def _after_edit(self, message: str) -> None:
        self.canvas.update()
        self.statusBar().showMessage(message)
        self._refresh_panel()

    def _on_hover_changed(self, pixel: tuple[int, int] | None) -> None:
        self.current_hover = pixel
        self._refresh_panel()

    def _on_selection_changed(self, rect: Rect | None) -> None:
        self.current_selection = rect
        if self.document is not None and rect is not None and rect.area == 1:
            r, g, b = self.document.get_pixel(rect.x, rect.y)
            for spin, value in ((self.pixel_r, r), (self.pixel_g, g), (self.pixel_b, b)):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self._refresh_panel()

    def _on_paste_target_changed(self, target: tuple[int, int] | None) -> None:
        if target is not None:
            self.statusBar().showMessage(f"Paste preview at ({target[0]}, {target[1]})")

    def _refresh_panel(self) -> None:
        has_doc = self.document is not None
        selection = self.current_selection
        self.path_label.setText(str(self.document.source_path) if has_doc else "No image loaded")
        self.size_label.setText(
            f"Size: {self.document.width} x {self.document.height}" if has_doc else "Size: -"
        )
        self.hover_label.setText(self._format_hover())
        self.selection_label.setText(self._format_selection())
        self.pixel_label.setText(self._format_selected_pixel())
        self.sample_label.setText(f"Sample: {self.sampled_rgb}" if self.sampled_rgb else "Sample: -")

        one_pixel = has_doc and selection is not None and selection.area == 1
        region = has_doc and selection is not None and selection.area > 0
        self.pixel_group.setEnabled(one_pixel)
        self.region_group.setEnabled(region)
        self.copy_button.setEnabled(region)
        self.paste_button.setEnabled(self.clipboard is not None)
        self.confirm_paste_button.setEnabled(self.canvas.paste_origin is not None)
        self.cancel_paste_button.setEnabled(self.canvas.paste_origin is not None)
        self.undo_action.setEnabled(has_doc and self.document.can_undo())
        self.redo_action.setEnabled(has_doc and self.document.can_redo())

    def _format_hover(self) -> str:
        if self.document is None or self.current_hover is None:
            return "Hover: -"
        x, y = self.current_hover
        return f"Hover: ({x}, {y}) RGB {self.document.get_pixel(x, y)}"

    def _format_selection(self) -> str:
        if self.current_selection is None:
            return "Selection: -"
        rect = self.current_selection
        return f"Selection: x={rect.x}, y={rect.y}, w={rect.width}, h={rect.height}"

    def _format_selected_pixel(self) -> str:
        if self.document is None or self.current_selection is None or self.current_selection.area != 1:
            return "Selected pixel: -"
        rect = self.current_selection
        return f"Selected pixel: ({rect.x}, {rect.y}) RGB {self.document.get_pixel(rect.x, rect.y)}"

    @staticmethod
    def _channel_spinbox() -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 255)
        return spin

    @staticmethod
    def _offset_spinbox() -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(-255, 255)
        return spin


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    app = QApplication(sys.argv[:1] + args)
    initial_image = Path(args[0]).expanduser() if args else None
    window = MainWindow(initial_image)
    window.show()
    return app.exec()


# Compatibility for callers that still import pixel_rgb_editor.main_window.
from .tabbed_main_window import MainWindow as MainWindow, main as main  # noqa: E402,F401
