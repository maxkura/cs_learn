"""Tabbed PySide6 main window for the pixel RGB editor."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .canvas import ImageCanvas
from .core import ClipboardRegion, RGBImageDocument, Rect


class EditorTab(QWidget):
    """One open image document with its own canvas and interaction state."""

    def __init__(self, document: RGBImageDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.document = document
        self.canvas = ImageCanvas()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.canvas.set_document(document)

    @property
    def title_base(self) -> str:
        if self.document.source_path is None:
            return "Untitled"
        return self.document.source_path.name

    @property
    def title(self) -> str:
        suffix = "*" if self.document.dirty else ""
        return f"{self.title_base}{suffix}"


class MainWindow(QMainWindow):
    """Multi-document editor window with a shared in-app RGB clipboard."""

    def __init__(self, initial_image: Path | None = None) -> None:
        super().__init__()
        self.clipboard: ClipboardRegion | None = None
        self.sampled_rgb: tuple[int, int, int] | None = None

        self.tabs = QTabWidget()
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
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tabs)
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
        self.open_action = open_action

        export_action = QAction("Export Copy...", self)
        export_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        export_action.triggered.connect(self.export_dialog)
        self.export_action = export_action

        close_tab_action = QAction("Close Tab", self)
        close_tab_action.setShortcut(QKeySequence.Close)
        close_tab_action.triggered.connect(self.close_current_tab)
        self.close_tab_action = close_tab_action

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
        self.copy_action = copy_action

        paste_action = QAction("Paste Preview", self)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.triggered.connect(self.start_paste_preview)
        self.paste_action = paste_action

        confirm_action = QAction("Confirm Paste", self)
        confirm_action.setShortcut(QKeySequence(Qt.Key_Return))
        confirm_action.triggered.connect(self.confirm_paste)
        self.confirm_action = confirm_action

        cancel_action = QAction("Cancel Paste", self)
        cancel_action.setShortcut(QKeySequence(Qt.Key_Escape))
        cancel_action.triggered.connect(self.cancel_paste)
        self.cancel_action = cancel_action

        toolbar = self.addToolBar("Main")
        for action in (
            open_action,
            export_action,
            close_tab_action,
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
        file_menu.addAction(close_tab_action)

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
        self.tabs.currentChanged.connect(self._on_current_tab_changed)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.apply_pixel_button.clicked.connect(self.apply_pixel_edit)
        self.apply_fixed_button.clicked.connect(self.apply_fixed_region)
        self.apply_offset_button.clicked.connect(self.apply_offset_region)
        self.sample_button.clicked.connect(self.sample_hover_pixel)
        self.fill_sample_button.clicked.connect(self.fill_selection_with_sample)
        self.copy_button.clicked.connect(self.copy_selection)
        self.paste_button.clicked.connect(self.start_paste_preview)
        self.confirm_paste_button.clicked.connect(self.confirm_paste)
        self.cancel_paste_button.clicked.connect(self.cancel_paste)

    def current_tab(self) -> EditorTab | None:
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, EditorTab) else None

    def current_document(self) -> RGBImageDocument | None:
        tab = self.current_tab()
        return tab.document if tab is not None else None

    def current_canvas(self) -> ImageCanvas | None:
        tab = self.current_tab()
        return tab.canvas if tab is not None else None

    def current_selection(self) -> Rect | None:
        canvas = self.current_canvas()
        return canvas.selection if canvas is not None else None

    def current_hover(self) -> tuple[int, int] | None:
        canvas = self.current_canvas()
        return canvas.hover_pixel if canvas is not None else None

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
            document = RGBImageDocument.open(path)
        except Exception as exc:  # pragma: no cover - UI guard
            QMessageBox.critical(self, "Open failed", str(exc))
            return

        tab = EditorTab(document)
        tab.canvas.hoverChanged.connect(self._on_hover_changed)
        tab.canvas.selectionChanged.connect(self._on_selection_changed)
        tab.canvas.pasteTargetChanged.connect(self._on_paste_target_changed)
        index = self.tabs.addTab(tab, tab.title)
        self.tabs.setCurrentIndex(index)
        self.statusBar().showMessage(f"Loaded {path}")
        self._refresh_panel()

    def export_dialog(self) -> None:
        document = self.current_document()
        if document is None:
            self.statusBar().showMessage("No image loaded")
            return
        saved = self._export_document(document)
        if saved is not None:
            self.statusBar().showMessage(f"Exported {saved}")
            self._refresh_panel()

    def close_current_tab(self) -> None:
        index = self.tabs.currentIndex()
        if index >= 0:
            self.close_tab(index)

    def close_tab(self, index: int) -> None:
        if not (0 <= index < self.tabs.count()):
            return
        self.tabs.setCurrentIndex(index)
        if not self._confirm_close_tab(index):
            return
        widget = self.tabs.widget(index)
        if isinstance(widget, EditorTab):
            widget.canvas.clear_paste_preview()
        self.tabs.removeTab(index)
        if widget is not None:
            widget.deleteLater()
        self._refresh_panel()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        for index in range(self.tabs.count()):
            self.tabs.setCurrentIndex(index)
            if not self._confirm_close_tab(index):
                event.ignore()
                return
        self._clear_all_paste_previews()
        event.accept()

    def apply_pixel_edit(self) -> None:
        document = self.current_document()
        selection = self.current_selection()
        if document is None or selection is None:
            return
        if selection.area != 1:
            self.statusBar().showMessage("Select exactly one pixel to edit it")
            return
        changed = document.set_pixel(
            selection.x,
            selection.y,
            (self.pixel_r.value(), self.pixel_g.value(), self.pixel_b.value()),
        )
        self._after_edit("Pixel updated" if changed else "Pixel already had that RGB")

    def apply_fixed_region(self) -> None:
        document = self.current_document()
        selection = self.current_selection()
        if document is None or selection is None:
            return
        changed = document.set_region(
            selection,
            (self.fixed_r.value(), self.fixed_g.value(), self.fixed_b.value()),
        )
        self._after_edit("Selection RGB updated" if changed else "Selection already matched")

    def apply_offset_region(self) -> None:
        document = self.current_document()
        selection = self.current_selection()
        if document is None or selection is None:
            return
        changed = document.offset_region(
            selection,
            (self.offset_r.value(), self.offset_g.value(), self.offset_b.value()),
        )
        self._after_edit("Selection RGB offset applied" if changed else "Offset changed nothing")

    def sample_hover_pixel(self) -> None:
        document = self.current_document()
        hover = self.current_hover()
        if document is None or hover is None:
            self.statusBar().showMessage("Hover over a pixel to sample it")
            return
        self.sampled_rgb = document.get_pixel(*hover)
        self._refresh_panel()
        self.statusBar().showMessage(f"Sampled RGB {self.sampled_rgb}")

    def fill_selection_with_sample(self) -> None:
        document = self.current_document()
        selection = self.current_selection()
        if document is None or selection is None or self.sampled_rgb is None:
            self.statusBar().showMessage("Need a selection and sampled RGB")
            return
        changed = document.set_region(selection, self.sampled_rgb)
        self._after_edit("Selection filled with sampled RGB" if changed else "Selection already matched sample")

    def copy_selection(self) -> None:
        document = self.current_document()
        selection = self.current_selection()
        if document is None or selection is None:
            self.statusBar().showMessage("No selection to copy")
            return
        self.clipboard = document.copy_region(selection)
        self.statusBar().showMessage(f"Copied {self.clipboard.width}x{self.clipboard.height} pixels")
        self._refresh_panel()

    def start_paste_preview(self) -> None:
        canvas = self.current_canvas()
        if canvas is None:
            self.statusBar().showMessage("No image loaded")
            return
        if self.clipboard is None:
            self.statusBar().showMessage("Copy a selection before pasting")
            return
        self._clear_all_paste_previews()
        canvas.begin_paste_preview(self.clipboard.width, self.clipboard.height)
        self.statusBar().showMessage("Move preview to target, then press Enter to confirm or Esc to cancel")
        self._refresh_panel()

    def confirm_paste(self) -> None:
        document = self.current_document()
        canvas = self.current_canvas()
        if document is None or canvas is None or self.clipboard is None or canvas.paste_origin is None:
            return
        rect = document.paste_region(self.clipboard, *canvas.paste_origin)
        canvas.clear_paste_preview()
        if rect is not None:
            canvas.set_selection(rect)
            self._after_edit(f"Pasted into {rect.width}x{rect.height} region")
        else:
            self.statusBar().showMessage("Paste target is outside the image")
            self._refresh_panel()

    def cancel_paste(self) -> None:
        canvas = self.current_canvas()
        if canvas is not None:
            canvas.clear_paste_preview()
        self.statusBar().showMessage("Paste canceled")
        self._refresh_panel()

    def undo(self) -> None:
        document = self.current_document()
        canvas = self.current_canvas()
        if document is None or canvas is None:
            return
        rect = document.undo()
        if rect is not None:
            canvas.set_selection(rect)
            self._after_edit("Undo")

    def redo(self) -> None:
        document = self.current_document()
        canvas = self.current_canvas()
        if document is None or canvas is None:
            return
        rect = document.redo()
        if rect is not None:
            canvas.set_selection(rect)
            self._after_edit("Redo")

    def _export_document(self, document: RGBImageDocument) -> Path | None:
        default_path = document.default_export_path()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Edited Copy",
            str(default_path),
            "PNG image (*.png)",
        )
        if not path:
            return None
        output = Path(path)
        if output.exists():
            answer = QMessageBox.question(
                self,
                "Overwrite existing file?",
                f"{output} already exists. Overwrite it?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return None
        try:
            saved = document.export_png(output)
        except Exception as exc:  # pragma: no cover - UI guard
            QMessageBox.critical(self, "Export failed", str(exc))
            return None
        self._refresh_all_tab_titles()
        return saved

    def _confirm_close_tab(self, index: int) -> bool:
        tab = self.tabs.widget(index)
        if not isinstance(tab, EditorTab) or not tab.document.dirty:
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Close edited image?")
        box.setText(f"{tab.title_base} has unsaved edits.")
        box.setInformativeText("Export an edited copy before closing?")
        save_button = box.addButton("Save/Export", QMessageBox.ButtonRole.AcceptRole)
        discard_button = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()

        clicked = box.clickedButton()
        if clicked == save_button:
            return self._export_document(tab.document) is not None
        if clicked == discard_button:
            return True
        if clicked == cancel_button:
            return False
        return False

    def _after_edit(self, message: str) -> None:
        canvas = self.current_canvas()
        if canvas is not None:
            canvas.update()
        self.statusBar().showMessage(message)
        self._refresh_all_tab_titles()
        self._refresh_panel()

    def _on_hover_changed(self, pixel: tuple[int, int] | None) -> None:
        if self.sender() is not self.current_canvas():
            return
        self._refresh_panel()

    def _on_selection_changed(self, rect: Rect | None) -> None:
        document = self.current_document()
        if self.sender() is not self.current_canvas():
            return
        if document is not None and rect is not None and rect.area == 1:
            r, g, b = document.get_pixel(rect.x, rect.y)
            for spin, value in ((self.pixel_r, r), (self.pixel_g, g), (self.pixel_b, b)):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self._refresh_panel()

    def _on_paste_target_changed(self, target: tuple[int, int] | None) -> None:
        if self.sender() is not self.current_canvas():
            return
        if target is not None:
            self.statusBar().showMessage(f"Paste preview at ({target[0]}, {target[1]})")
        self._refresh_panel()

    def _on_current_tab_changed(self, index: int) -> None:
        self._clear_all_paste_previews()
        self._refresh_panel()
        tab = self.current_tab()
        if tab is None:
            self.setWindowTitle("Pixel RGB Editor")
        else:
            self.setWindowTitle(f"Pixel RGB Editor - {tab.title_base}")

    def _refresh_panel(self) -> None:
        document = self.current_document()
        canvas = self.current_canvas()
        selection = canvas.selection if canvas is not None else None
        paste_origin = canvas.paste_origin if canvas is not None else None
        has_doc = document is not None

        self.path_label.setText(str(document.source_path) if has_doc else "No image loaded")
        self.size_label.setText(
            f"Size: {document.width} x {document.height}" if has_doc else "Size: -"
        )
        self.hover_label.setText(self._format_hover())
        self.selection_label.setText(self._format_selection())
        self.pixel_label.setText(self._format_selected_pixel())
        self.sample_label.setText(f"Sample: {self.sampled_rgb}" if self.sampled_rgb else "Sample: -")

        one_pixel = has_doc and selection is not None and selection.area == 1
        region = has_doc and selection is not None and selection.area > 0
        has_preview = has_doc and paste_origin is not None

        self.pixel_group.setEnabled(one_pixel)
        self.region_group.setEnabled(region)
        self.copy_button.setEnabled(region)
        self.paste_button.setEnabled(has_doc and self.clipboard is not None)
        self.confirm_paste_button.setEnabled(has_preview)
        self.cancel_paste_button.setEnabled(has_preview)

        self.export_action.setEnabled(has_doc)
        self.close_tab_action.setEnabled(has_doc)
        self.copy_action.setEnabled(region)
        self.paste_action.setEnabled(has_doc and self.clipboard is not None)
        self.confirm_action.setEnabled(has_preview)
        self.cancel_action.setEnabled(has_preview)
        self.undo_action.setEnabled(has_doc and document.can_undo())
        self.redo_action.setEnabled(has_doc and document.can_redo())

    def _format_hover(self) -> str:
        document = self.current_document()
        hover = self.current_hover()
        if document is None or hover is None:
            return "Hover: -"
        x, y = hover
        return f"Hover: ({x}, {y}) RGB {document.get_pixel(x, y)}"

    def _format_selection(self) -> str:
        selection = self.current_selection()
        if selection is None:
            return "Selection: -"
        return f"Selection: x={selection.x}, y={selection.y}, w={selection.width}, h={selection.height}"

    def _format_selected_pixel(self) -> str:
        document = self.current_document()
        selection = self.current_selection()
        if document is None or selection is None or selection.area != 1:
            return "Selected pixel: -"
        return f"Selected pixel: ({selection.x}, {selection.y}) RGB {document.get_pixel(selection.x, selection.y)}"

    def _clear_all_paste_previews(self) -> None:
        for index in range(self.tabs.count()):
            tab = self.tabs.widget(index)
            if isinstance(tab, EditorTab):
                tab.canvas.clear_paste_preview()

    def _refresh_all_tab_titles(self) -> None:
        for index in range(self.tabs.count()):
            tab = self.tabs.widget(index)
            if isinstance(tab, EditorTab):
                self.tabs.setTabText(index, tab.title)

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
