"""PyQt5 интерфейс для просмотра дерева каталогов.""" 

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from tree_core import (
    DEFAULT_DIR_IGNORES,
    DEFAULT_FILE_IGNORES,
    DirectoryNode,
    ScanConfig,
    ScanFilters,
    render_tree,
    scan_tree,
)


MAX_SEARCH_RESULTS = 500


@dataclass
class GUIState:
    include_files: bool = True
    include_hidden: bool = False
    max_depth: Optional[int] = 3
    include_patterns: Sequence[str] = ()
    exclude_patterns: Sequence[str] = ()
    skip_empty_dirs: bool = False
    dir_ignores: Sequence[str] = DEFAULT_DIR_IGNORES
    file_ignores: Sequence[str] = DEFAULT_FILE_IGNORES


class ScanWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, float)
    error = QtCore.pyqtSignal(str)

    def __init__(self, config: ScanConfig) -> None:
        super().__init__()
        self._config = config

    @QtCore.pyqtSlot()
    def run(self) -> None:
        start = time.perf_counter()
        try:
            node = scan_tree(self._config)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
            return
        elapsed = time.perf_counter() - start
        self.finished.emit(node, elapsed)


class TreeViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, root: Path, state: GUIState) -> None:
        super().__init__()
        self.setWindowTitle("Directory Explorer")
        self.resize(1100, 720)

        self._current_root = root
        self._gui_state = state
        self._current_node: Optional[DirectoryNode] = None
        self._scanning = False
        self._worker_thread: Optional[QtCore.QThread] = None
        self._worker: Optional[ScanWorker] = None
        self._search_matches: List[DirectoryNode] = []
        self._search_query: str = ""

        style = self.style()
        self._folder_icon = style.standardIcon(QtWidgets.QStyle.SP_DirIcon)
        self._file_icon = style.standardIcon(QtWidgets.QStyle.SP_FileIcon)

        self._build_ui()
        self._apply_initial_state()
        self.load_path(root)

    # UI ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        path_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Корневая директория")
        self.path_edit.returnPressed.connect(self._on_path_entered)
        path_row.addWidget(QtWidgets.QLabel("Путь:"))
        path_row.addWidget(self.path_edit, stretch=1)

        browse_btn = QtWidgets.QPushButton("Выбрать…")
        browse_btn.clicked.connect(self._browse_for_directory)
        path_row.addWidget(browse_btn)

        layout.addLayout(path_row)

        filters_row = QtWidgets.QHBoxLayout()
        self.include_files_cb = QtWidgets.QCheckBox("Файлы")
        filters_row.addWidget(self.include_files_cb)

        self.hidden_cb = QtWidgets.QCheckBox("Скрытые")
        filters_row.addWidget(self.hidden_cb)

        self.skip_empty_cb = QtWidgets.QCheckBox("Пропускать пустые")
        filters_row.addWidget(self.skip_empty_cb)

        filters_row.addWidget(QtWidgets.QLabel("Глубина:"))
        self.depth_spin = QtWidgets.QSpinBox()
        self.depth_spin.setRange(0, 64)
        self.depth_spin.setSpecialValueText("Все")
        filters_row.addWidget(self.depth_spin)

        filters_row.addWidget(QtWidgets.QLabel("Include:"))
        self.include_edit = QtWidgets.QLineEdit()
        self.include_edit.setPlaceholderText("*.py, src/*")
        filters_row.addWidget(self.include_edit, stretch=1)

        filters_row.addWidget(QtWidgets.QLabel("Exclude:"))
        self.exclude_edit = QtWidgets.QLineEdit()
        self.exclude_edit.setPlaceholderText("tests/*")
        filters_row.addWidget(self.exclude_edit, stretch=1)

        apply_btn = QtWidgets.QPushButton("Применить")
        apply_btn.clicked.connect(lambda: self.load_path(self._current_root))
        filters_row.addWidget(apply_btn)

        layout.addLayout(filters_row)

        toolbar_row = QtWidgets.QHBoxLayout()
        self.back_btn = QtWidgets.QPushButton("Назад")
        self.back_btn.clicked.connect(self._navigate_up)
        toolbar_row.addWidget(self.back_btn)

        refresh_btn = QtWidgets.QPushButton("Обновить")
        refresh_btn.clicked.connect(lambda: self.load_path(self._current_root))
        toolbar_row.addWidget(refresh_btn)

        export_btn = QtWidgets.QPushButton("Экспорт…")
        export_btn.clicked.connect(self._export_text)
        toolbar_row.addWidget(export_btn)

        toolbar_row.addStretch(1)
        layout.addLayout(toolbar_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.tree_widget = QtWidgets.QTreeWidget()
        self.tree_widget.setHeaderLabels(["Имя", "Тип", "Детали"])
        header = self.tree_widget.header()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.tree_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree_widget.itemSelectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self.tree_widget)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_panel.setLayout(right_layout)

        search_container = QtWidgets.QWidget()
        search_container_layout = QtWidgets.QVBoxLayout()
        search_container_layout.setContentsMargins(0, 0, 0, 0)
        search_container_layout.addWidget(QtWidgets.QLabel("Поиск"))

        search_row = QtWidgets.QHBoxLayout()
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Введите часть имени…")
        self.search_edit.returnPressed.connect(self._run_search)
        search_row.addWidget(self.search_edit, stretch=1)

        search_btn = QtWidgets.QPushButton("Найти")
        search_btn.clicked.connect(self._run_search)
        search_row.addWidget(search_btn)

        clear_btn = QtWidgets.QPushButton("Сбросить")
        clear_btn.clicked.connect(self._clear_search)
        search_row.addWidget(clear_btn)

        search_container_layout.addLayout(search_row)
        search_container.setLayout(search_container_layout)
        right_layout.addWidget(search_container)

        self.search_results = QtWidgets.QListWidget()
        self.search_results.itemSelectionChanged.connect(self._on_search_selection_changed)
        self.search_results.itemDoubleClicked.connect(self._on_search_item_double_clicked)
        right_layout.addWidget(self.search_results)

        self.text_view = QtWidgets.QPlainTextEdit()
        self.text_view.setReadOnly(True)
        right_layout.addWidget(self.text_view)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Готово")


    def _apply_initial_state(self) -> None:
        self.include_files_cb.setChecked(self._gui_state.include_files)
        self.hidden_cb.setChecked(self._gui_state.include_hidden)
        self.skip_empty_cb.setChecked(self._gui_state.skip_empty_dirs)
        self.depth_spin.setValue(self._gui_state.max_depth or 0)
        self.include_edit.setText(", ".join(self._gui_state.include_patterns))
        self.exclude_edit.setText(", ".join(self._gui_state.exclude_patterns))
        self.path_edit.setText(str(self._current_root))
        self._update_back_button()

    def _update_tree_text(self) -> None:
        if self._current_node:
            self.text_view.setPlainText(render_tree(self._current_node).rstrip())
        else:
            self.text_view.clear()

    def _run_search(self, silent: bool = False) -> None:
        if not self._current_node:
            self.status_bar.showMessage("Дерево ещё не загружено")
            return
        query = self.search_edit.text().strip()
        self._search_query = query
        self.search_results.clear()
        self._search_matches = []
        if not query:
            self._update_tree_text()
            if not silent:
                self.status_bar.showMessage("Поиск очищен")
            return
        matches, truncated = self._collect_matches(query)
        self._search_matches = matches
        base = self._current_root
        for node in matches:
            prefix = "[DIR]" if node.is_dir else "[FILE]"
            rel = self._relative_path(base, Path(node.path))
            label = f"{prefix} {rel}"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, node)
            item.setToolTip(node.path)
            self.search_results.addItem(item)
        if matches:
            self.search_results.setCurrentRow(0)
            if not silent:
                message = f"Найдено совпадений: {len(matches)}"
                if truncated:
                    message += f" (показаны первые {MAX_SEARCH_RESULTS})"
                self.status_bar.showMessage(message)
        else:
            self.text_view.setPlainText("Совпадений не найдено")
            if not silent:
                self.status_bar.showMessage("Совпадений не найдено")

    def _clear_search(self) -> None:
        self.search_edit.clear()
        self._search_query = ""
        self.search_results.clear()
        self._search_matches = []
        self._update_tree_text()
        self.status_bar.showMessage("Поиск очищен")

    def _collect_matches(self, term: str) -> Tuple[List[DirectoryNode], bool]:
        if not self._current_node:
            return [], False
        limit = MAX_SEARCH_RESULTS
        results: List[DirectoryNode] = []
        term_lower = term.lower()
        truncated = False

        def visit(node: DirectoryNode) -> None:
            nonlocal limit, truncated
            if limit == 0:
                truncated = True
                return
            if term_lower in node.name.lower():
                results.append(node)
                limit -= 1
                if limit == 0:
                    truncated = True
                    return
            for child in node.children:
                visit(child)

        visit(self._current_node)
        return results, truncated

    def _relative_path(self, base: Path, target: Path) -> str:
        try:
            rel = target.resolve().relative_to(base.resolve())
            text = str(rel) if str(rel) else "."
        except Exception:
            text = str(target)
        return text.replace("\\", "/")

    def _normalize_path(self, path: str) -> str:
        try:
            return str(Path(path).resolve())
        except Exception:
            return os.path.normpath(path)

    def _display_node_details(self, node: DirectoryNode) -> None:
        if node.is_dir:
            self.text_view.setPlainText(render_tree(node).rstrip())
            return
        lines = [f"Путь: {node.path}"]
        if node.size is not None:
            lines.append(f"Размер: {_format_size(node.size)}")
        if node.mtime is not None:
            try:
                ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(node.mtime))
                lines.append(f"Изменено: {ts}")
            except (ValueError, OSError):
                pass
        self.text_view.setPlainText(" ".join(lines))

    def _focus_tree_item(self, path: str) -> None:
        target = self._normalize_path(path)
        if self.tree_widget.topLevelItemCount() == 0:
            return
        stack: List[QtWidgets.QTreeWidgetItem] = [self.tree_widget.topLevelItem(i) for i in range(self.tree_widget.topLevelItemCount())]
        while stack:
            item = stack.pop()
            item_path = item.data(0, QtCore.Qt.UserRole)
            if item_path and self._normalize_path(str(item_path)) == target:
                parent = item.parent()
                while parent:
                    parent.setExpanded(True)
                    parent = parent.parent()
                self.tree_widget.setCurrentItem(item)
                self.tree_widget.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
                return
            for i in range(item.childCount() - 1, -1, -1):
                stack.append(item.child(i))

    def _on_search_selection_changed(self) -> None:
        items = self.search_results.selectedItems()
        if not items:
            return
        node = items[0].data(QtCore.Qt.UserRole)
        if not isinstance(node, DirectoryNode):
            return
        self._display_node_details(node)
        self._focus_tree_item(node.path)

    def _on_search_item_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        node = item.data(QtCore.Qt.UserRole)
        if not isinstance(node, DirectoryNode):
            return
        if node.is_dir:
            self.load_path(Path(node.path))
        else:
            self._focus_tree_item(node.path)

    # События ---------------------------------------------------------------
    def _on_path_entered(self) -> None:
        text = self.path_edit.text().strip()
        if not text:
            return
        self.load_path(Path(text))

    def _browse_for_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Выбор директории", str(self._current_root)
        )
        if directory:
            self.load_path(Path(directory))

    def _navigate_up(self) -> None:
        parent = self._current_root.parent
        if parent == self._current_root:
            return
        self.load_path(parent)

    def _on_item_double_clicked(self, item: QtWidgets.QTreeWidgetItem) -> None:
        path = item.data(0, QtCore.Qt.UserRole)
        if not path:
            return
        file_path = Path(path)
        if file_path.is_dir():
            self.load_path(file_path)

    def _on_selection_changed(self) -> None:
        items = self.tree_widget.selectedItems()
        if not items:
            return
        path = items[0].data(0, QtCore.Qt.UserRole)
        if path:
            self.status_bar.showMessage(path)

    def _export_text(self) -> None:
        if not self._current_node:
            QtWidgets.QMessageBox.warning(self, "Экспорт", "Нечего экспортировать")
            return
        suggested = self._current_root / "PROJECT_STRUCTURE.txt"
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Сохранить дерево",
            str(suggested),
            "Text files (*.txt);;All files (*.*)",
        )
        if not filename:
            return
        Path(filename).write_text(render_tree(self._current_node), encoding="utf-8-sig")
        self.status_bar.showMessage(f"Сохранено в {filename}")

    # Загрузка --------------------------------------------------------------
    def load_path(self, path: Path) -> None:
        if self._scanning:
            return
        resolved = path.expanduser().resolve()
        if not resolved.exists() or not resolved.is_dir():
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Директория не найдена: {resolved}")
            return
        self._current_root = resolved
        self.path_edit.setText(str(resolved))
        self._update_back_button()

        filters = self._build_filters()
        config = ScanConfig(root=str(resolved), filters=filters)
        self._start_scan(config)

    def _build_filters(self) -> ScanFilters:
        include_files = self.include_files_cb.isChecked()
        include_hidden = self.hidden_cb.isChecked()
        max_depth_value = self.depth_spin.value()
        include_patterns = _split_patterns(self.include_edit.text())
        exclude_patterns = _split_patterns(self.exclude_edit.text())

        self._gui_state = replace(
            self._gui_state,
            include_files=include_files,
            include_hidden=include_hidden,
            max_depth=max_depth_value or None,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            skip_empty_dirs=self.skip_empty_cb.isChecked(),
        )

        return ScanFilters(
            include_files=include_files,
            include_hidden=include_hidden,
            max_depth=self._gui_state.max_depth,
            dir_ignores=tuple(self._gui_state.dir_ignores),
            file_ignores=tuple(self._gui_state.file_ignores),
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            skip_empty_dirs=self._gui_state.skip_empty_dirs,
        )

    def _start_scan(self, config: ScanConfig) -> None:
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait()
            self._worker_thread = None
            self._worker = None

        self._scanning = True
        self.status_bar.showMessage("Сканирование…")
        self.tree_widget.setEnabled(False)
        self.text_view.clear()

        worker = ScanWorker(config)
        thread = QtCore.QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_scan_finished)
        worker.error.connect(self._on_scan_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.start()

        self._worker_thread = thread
        self._worker = worker

    @QtCore.pyqtSlot()
    def _on_thread_finished(self) -> None:
        self._worker_thread = None
        self._worker = None

    @QtCore.pyqtSlot(object, float)
    def _on_scan_finished(self, node: DirectoryNode, elapsed: float) -> None:
        self._scanning = False
        self.tree_widget.setEnabled(True)
        self._current_node = node
        self._populate_tree(node)
        if self._search_query:
            self._run_search(silent=True)
        else:
            self.search_results.clear()
            self._search_matches = []
            self._update_tree_text()
        self.status_bar.showMessage(f"Загружено за {elapsed:.2f} c")

    @QtCore.pyqtSlot(str)
    def _on_scan_error(self, message: str) -> None:
        self._scanning = False
        self.tree_widget.setEnabled(True)
        self.search_results.clear()
        self._search_matches = []
        QtWidgets.QMessageBox.critical(self, "Ошибка сканирования", message)
        self.status_bar.showMessage("Ошибка")

    def _populate_tree(self, node: DirectoryNode) -> None:
        self.tree_widget.clear()
        root_item = self._make_item(node)
        self.tree_widget.addTopLevelItem(root_item)
        root_item.setExpanded(True)
        self.tree_widget.scrollToTop()

    def _make_item(self, node: DirectoryNode) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem()
        label = node.name or node.path
        item.setText(0, label)
        item.setData(0, QtCore.Qt.UserRole, node.path)
        if node.is_dir:
            item.setIcon(0, self._folder_icon)
            item.setText(1, "dir")
            item.setText(2, f"{len(node.children)} элементов")
            for child in node.children:
                item.addChild(self._make_item(child))
        else:
            item.setIcon(0, self._file_icon)
            item.setText(1, "file")
            item.setText(2, _format_file_details(node))
        return item

    def _update_back_button(self) -> None:
        parent = self._current_root.parent
        self.back_btn.setEnabled(parent != self._current_root)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait()
            self._worker_thread = None
        self._worker = None
        self._scanning = False
        super().closeEvent(event)


def launch_gui(start_path: Optional[Path], options: Dict[str, object]) -> int:
    app = QtWidgets.QApplication.instance()
    owns_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv or ["make_gui"])
        owns_app = True

    _apply_dark_theme(app)

    root = (start_path or Path.cwd()).expanduser().resolve()
    initial_max_depth = options.get("max_depth")
    gui_state = GUIState(
        include_files=bool(options.get("include_files", True)),
        include_hidden=bool(options.get("include_hidden", False)),
        max_depth=3 if initial_max_depth is None else initial_max_depth,
        include_patterns=tuple(options.get("include_patterns", ())),
        exclude_patterns=tuple(options.get("exclude_patterns", ())),
        skip_empty_dirs=bool(options.get("skip_empty_dirs", False)),
        dir_ignores=tuple(options.get("dir_ignores", DEFAULT_DIR_IGNORES)),
        file_ignores=tuple(options.get("file_ignores", DEFAULT_FILE_IGNORES)),
    )

    window = TreeViewerWindow(root, gui_state)
    window.show()

    if owns_app:
        return app.exec_()
    return 0


def _apply_dark_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 32))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(225, 225, 225))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(24, 24, 26))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(36, 36, 38))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(225, 225, 225))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 47, 51))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(225, 225, 225))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(76, 140, 199))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(150, 150, 150))
    app.setPalette(palette)
    app.setStyleSheet(
        """
        QWidget { color: #E1E1E1; }
        QLineEdit, QPlainTextEdit, QListWidget, QTreeWidget { background-color: #1F1F22; color: #E1E1E1; border: 1px solid #3C3F41; }
        QTreeWidget::item:selected, QListWidget::item:selected { background-color: #4C8CC7; color: #FFFFFF; }
        QPushButton { background-color: #3D85C6; color: #FFFFFF; border-radius: 4px; padding: 5px 10px; }
        QPushButton:hover { background-color: #4F97D6; }
        QPushButton:pressed { background-color: #34699D; }
        QCheckBox, QLabel { color: #E1E1E1; }
        QStatusBar { color: #E1E1E1; }
        QSplitter::handle { background-color: #3C3F41; }
        QScrollBar:vertical { background: #2B2B2B; width: 12px; margin: 0; }
        QScrollBar::handle:vertical { background: #4C8CC7; min-height: 20px; border-radius: 4px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """
    )



def _split_patterns(text: str) -> Tuple[str, ...]:
    if not text:
        return ()
    parts = [part.strip() for part in text.split(",")]
    return tuple(part for part in parts if part)


def _format_file_details(node: DirectoryNode) -> str:
    if node.size is None:
        return ""
    return _format_size(node.size)


def _format_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"

def _parse_cli(argv: Optional[Sequence[str]] = None) -> Tuple[Optional[Path], Dict[str, object]]:
    parser = argparse.ArgumentParser(description="Запуск PyQt5 проводника.")
    parser.add_argument("path", nargs="?", default=None, help="Стартовая директория.")
    parser.add_argument("--include-hidden", action="store_true", help="Показывать скрытые элементы.")
    parser.add_argument("--dirs-only", action="store_true", help="Не показывать файлы.")
    parser.add_argument("--max-depth", type=int, default=None, help="Ограничение по глубине.")
    parser.add_argument("--include-pattern", action="append", default=None, help="Include-глоб (можно повторять).")
    parser.add_argument("--exclude-pattern", action="append", default=None, help="Exclude-глоб (можно повторять).")
    parser.add_argument("--skip-empty-dirs", action="store_true", help="Пропускать пустые каталоги при include.")

    args = parser.parse_args(argv)
    start_path = Path(args.path).expanduser() if args.path else None
    options = {
        "include_hidden": args.include_hidden,
        "include_files": not args.dirs_only,
        "max_depth": args.max_depth,
        "include_patterns": tuple(args.include_pattern or ()),
        "exclude_patterns": tuple(args.exclude_pattern or ()),
        "skip_empty_dirs": args.skip_empty_dirs,
    }
    return start_path, options


if __name__ == "__main__":
    start, opts = _parse_cli()
    raise SystemExit(launch_gui(start, opts))

