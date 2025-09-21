"""Единый инструмент структуры каталогов: CLI, PyQt5 GUI, MCP."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from tree_core import (
    DEFAULT_DIR_IGNORES,
    DEFAULT_FILE_IGNORES,
    ScanConfig,
    ScanFilters,
    render_tree,
    scan_tree,
    search_directory,
)

DEFAULT_OUTPUT_NAME = "PROJECT_STRUCTURE.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make.py",
        description="Просмотр структуры каталогов с запуском из CLI, GUI и MCP.",
    )
    subparsers = parser.add_subparsers(dest="command")

    export = subparsers.add_parser("export", help="Создать текстовый снимок дерева.")
    export.add_argument("path", help="Корневая директория.")
    export.add_argument("--name", default=DEFAULT_OUTPUT_NAME, help="Имя выходного файла.")
    export.add_argument("--max-depth", type=int, default=None, help="Глубина обхода (0 = только корень).")
    export.add_argument("--dirs-only", action="store_true", help="Показывать только директории.")
    export.add_argument("--include-hidden", action="store_true", help="Включать скрытые файлы и папки.")
    export.add_argument("--no-ignore", action="store_true", help="Отключить стандартные игноры.")
    export.add_argument("--dir-ignore", action="append", default=None, help="Доп. директория для игнора.")
    export.add_argument("--file-ignore", action="append", default=None, help="Доп. файл/расширение для игнора.")
    export.add_argument("--include-pattern", action="append", default=None, help="Глоб для включения (можно перечислением).")
    export.add_argument("--exclude-pattern", action="append", default=None, help="Глоб для исключения.")
    export.add_argument("--skip-empty-dirs", action="store_true", help="Убирать пустые каталоги при фильтрах.")
    export.add_argument("--stdout", action="store_true", help="Продублировать вывод в консоль.")
    export.add_argument("--no-open", action="store_true", help="Не открывать Проводник после записи.")
    export.set_defaults(handler=run_export)


    search = subparsers.add_parser("search", help="Поиск файлов и директорий по имени.")
    search.add_argument("path", help="Корневая директория.")
    search.add_argument("query", help="Строка поиска (регистр не важен).")
    search.add_argument("--max-depth", type=int, default=None, help="Глубина обхода (0 = только корень).")
    search.add_argument("--dirs-only", action="store_true", help="Искать только директории.")
    search.add_argument("--include-hidden", action="store_true", help="Учитывать скрытые файлы и папки.")
    search.add_argument("--no-ignore", action="store_true", help="Отключить стандартные игноры.")
    search.add_argument("--dir-ignore", action="append", default=None, help="Доп. директория для игнора.")
    search.add_argument("--file-ignore", action="append", default=None, help="Доп. файл/расширение для игнора.")
    search.add_argument("--include-pattern", action="append", default=None, help="Include-глоб (можно перечислением).")
    search.add_argument("--exclude-pattern", action="append", default=None, help="Exclude-глоб (можно перечислением).")
    search.add_argument("--skip-empty-dirs", action="store_true", help="Пропускать пустые каталоги при include.")
    search.add_argument("--limit", type=int, default=100, help="Максимум совпадений (0 = без лимита).")
    search.add_argument("--show-tree", action="store_true", help="Для директорий выводить их дочернюю структуру.")
    search.set_defaults(handler=run_search)
    gui = subparsers.add_parser("gui", help="Запуск PyQt5 интерфейса.")
    gui.add_argument("path", nargs="?", default=None, help="Стартовая директория.")
    gui.add_argument("--include-hidden", action="store_true", help="Показывать скрытые по умолчанию.")
    gui.add_argument("--dirs-only", action="store_true", help="Начать только с директорий.")
    gui.add_argument("--max-depth", type=int, default=None, help="Начальная глубина.")
    gui.add_argument("--include-pattern", action="append", default=None, help="Начальные include-глобы.")
    gui.add_argument("--exclude-pattern", action="append", default=None, help="Начальные exclude-глобы.")
    gui.add_argument("--skip-empty-dirs", action="store_true", help="Не показывать пустые папки при include.")
    gui.set_defaults(handler=run_gui)

    mcp = subparsers.add_parser("mcp", help="Поднять MCP сервер.")
    mcp.add_argument("--host", default="127.0.0.1", help="Хост.")
    mcp.add_argument("--port", type=int, default=8765, help="Порт.")
    mcp.add_argument("--debug", action="store_true", help="Более подробный лог.")
    mcp.set_defaults(handler=run_mcp)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    normalized = _normalize_argv(argv)
    args = parser.parse_args(normalized)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


def run_export(args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser()
    if not root.exists():
        raise SystemExit(f"Путь не найден: {root}")
    if not root.is_dir():
        raise SystemExit(f"Это не директория: {root}")

    filters = _make_filters(
        include_files=not args.dirs_only,
        include_hidden=args.include_hidden,
        max_depth=args.max_depth,
        skip_empty=args.skip_empty_dirs,
        include_patterns=args.include_pattern,
        exclude_patterns=args.exclude_pattern,
        extra_dir_ignores=args.dir_ignore,
        extra_file_ignores=args.file_ignore,
        disable_defaults=args.no_ignore,
    )
    config = ScanConfig(root=str(root.resolve()), filters=filters)
    tree = scan_tree(config)
    output_text = render_tree(tree)

    out_file = write_text_file(root, args.name, output_text)
    print(f"Структура записана: {out_file}")
    if args.stdout:
        print()
        print(output_text)

    if not args.no_open:
        reveal_in_file_manager(out_file)
    return 0



def run_search(args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser()
    if not root.exists():
        raise SystemExit(f"Путь не найден: {root}")
    if not root.is_dir():
        raise SystemExit(f"Это не директория: {root}")

    filters = _make_filters(
        include_files=not args.dirs_only,
        include_hidden=args.include_hidden,
        max_depth=args.max_depth,
        skip_empty=args.skip_empty_dirs,
        include_patterns=args.include_pattern,
        exclude_patterns=args.exclude_pattern,
        extra_dir_ignores=args.dir_ignore,
        extra_file_ignores=args.file_ignore,
        disable_defaults=args.no_ignore,
    )
    config = ScanConfig(root=str(root.resolve()), filters=filters)
    limit_value = None if args.limit <= 0 else args.limit
    tree, matches, truncated = search_directory(config, args.query, limit=limit_value)

    if not matches:
        print("Совпадений не найдено")
        return 0

    base_path = Path(tree.path)
    for index, node in enumerate(matches, start=1):
        rel = _relative_path(base_path, Path(node.path))
        entry_type = "DIR" if node.is_dir else "FILE"
        print(f"[{index}] {entry_type}: {rel}")
        if args.show_tree and node.is_dir:
            print(render_tree(node).rstrip())
            print()
    total_msg = f"Всего совпадений: {len(matches)}"
    if truncated and limit_value is not None:
        total_msg += f" (показаны первые {limit_value})"
    print(total_msg)
    return 0


def _relative_path(base: Path, target: Path) -> str:
    try:
        rel = target.resolve().relative_to(base.resolve())
        text = str(rel) if str(rel) else "."
    except ValueError:
        text = str(target)
    return text


def run_gui(args: argparse.Namespace) -> int:
    try:
        from gui_app import launch_gui
    except ImportError as exc:
        raise SystemExit("Для GUI нужен PyQt5: pip install pyqt5") from exc

    start_path = Path(args.path).expanduser() if args.path else None
    options = {
        "include_hidden": args.include_hidden,
        "include_files": not args.dirs_only,
        "max_depth": args.max_depth,
        "include_patterns": _normalize_patterns(args.include_pattern),
        "exclude_patterns": _normalize_patterns(args.exclude_pattern),
        "skip_empty_dirs": args.skip_empty_dirs,
    }
    return launch_gui(start_path, options)


def run_mcp(args: argparse.Namespace) -> int:
    try:
        from mcp_server import serve
    except ImportError as exc:
        raise SystemExit("Для MCP режима нужен пакет fastmcp: pip install fastmcp") from exc
    return serve(host=args.host, port=args.port, debug=args.debug)


def write_text_file(root: Path, filename: str, content: str) -> Path:
    output_path = root / filename
    output_path.write_text(content, encoding="utf-8-sig")
    return output_path


def reveal_in_file_manager(path: Path) -> None:
    system = platform.system().lower()
    target = str(path.resolve())
    if "windows" in system:
        subprocess.run(["explorer", "/select,", target], check=False)
    elif "darwin" in system:
        subprocess.run(["open", "-R", target], check=False)
    else:
        folder = str(path.parent.resolve()) if path.is_file() else target
        subprocess.run(["xdg-open", folder], check=False)


def _normalize_argv(argv: Sequence[str]) -> List[str]:
    if not argv:
        return list(argv)
    commands = {"export", "gui", "mcp", "search"}
    first = argv[0]
    if first in commands or first.startswith("-"):
        return list(argv)
    return ["export", *argv]


def _normalize_patterns(values: Optional[Iterable[str]]) -> tuple:
    if not values:
        return ()
    normalized: List[str] = []
    for value in values:
        if not value:
            continue
        parts = [part.strip() for part in value.split(",")]
        normalized.extend(part for part in parts if part)
    return tuple(dict.fromkeys(normalized))


def _make_filters(
    *,
    include_files: bool,
    include_hidden: bool,
    max_depth: Optional[int],
    skip_empty: bool,
    include_patterns: Optional[Iterable[str]],
    exclude_patterns: Optional[Iterable[str]],
    extra_dir_ignores: Optional[Iterable[str]],
    extra_file_ignores: Optional[Iterable[str]],
    disable_defaults: bool,
) -> ScanFilters:
    dir_ignores = [] if disable_defaults else list(DEFAULT_DIR_IGNORES)
    file_ignores = [] if disable_defaults else list(DEFAULT_FILE_IGNORES)
    if extra_dir_ignores:
        dir_ignores.extend(extra_dir_ignores)
    if extra_file_ignores:
        file_ignores.extend(extra_file_ignores)

    return ScanFilters(
        include_files=include_files,
        include_hidden=include_hidden,
        max_depth=max_depth,
        dir_ignores=tuple(dict.fromkeys(dir_ignores)),
        file_ignores=tuple(dict.fromkeys(file_ignores)),
        include_patterns=_normalize_patterns(include_patterns),
        exclude_patterns=_normalize_patterns(exclude_patterns),
        skip_empty_dirs=skip_empty,
    )


if __name__ == "__main__":
    raise SystemExit(main())

