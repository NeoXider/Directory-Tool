"""FastMCP сервер для предоставления структуры каталогов без кэширования."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from fastmcp import FastMCP

from tree_core import (
    DEFAULT_DIR_IGNORES,
    DEFAULT_FILE_IGNORES,
    DirectoryNode,
    ScanConfig,
    ScanFilters,
    render_tree,
    scan_tree,
    search_directory,
)

mcp = FastMCP(
    "Directory Explorer MCP",
    instructions=(
        "Инструменты для просмотра структуры каталогов, выдачи списков файлов и текстовых деревьев."
    ),
)


@dataclass
class TreeStats:
    directories: int
    files: int
    depth: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "directories": self.directories,
            "files": self.files,
            "max_depth": self.depth,
        }


def _list_structure_impl(
    root: str,
    max_depth: Optional[int] = None,
    include_files: bool = True,
    include_hidden: bool = False,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    skip_empty_dirs: bool = False,
    no_ignore: bool = False,
    extra_dir_ignores: Optional[List[str]] = None,
    extra_file_ignores: Optional[List[str]] = None,
    include_json: bool = True,
) -> Dict:
    """Собрать дерево файловой системы для переданного каталога."""
    filters = _build_filters(
        include_files=include_files,
        include_hidden=include_hidden,
        max_depth=max_depth,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        skip_empty_dirs=skip_empty_dirs,
        no_ignore=no_ignore,
        extra_dir_ignores=extra_dir_ignores,
        extra_file_ignores=extra_file_ignores,
    )
    node, stats = _scan_tree(root, filters)
    response: Dict[str, object] = {
        "root": node.path,
        "summary": stats.to_dict(),
        "tree_text": render_tree(node).rstrip(),
    }
    if include_json:
        response["tree"] = node.to_dict()
    return response

@mcp.tool(
    name="list_structure",
    description="Вернуть дерево каталога c учётом фильтров и настроек обхода.",
)
def list_structure_tool(
    root: str,
    max_depth: Optional[int] = None,
    include_files: bool = True,
    include_hidden: bool = False,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    skip_empty_dirs: bool = False,
    no_ignore: bool = False,
    extra_dir_ignores: Optional[List[str]] = None,
    extra_file_ignores: Optional[List[str]] = None,
    include_json: bool = True,
) -> Dict:
    """Собрать дерево файловой системы для переданного каталога.

    Args:
        root: Абсолютный или относительный путь к директории.
        max_depth: Ограничение по глубине (``None`` — без ограничения).
        include_files: Включать ли файлы в дерево.
        include_hidden: Показывать ли скрытые элементы.
        include_patterns: Glob-шаблоны, которые должны совпасть, чтобы элемент попал в дерево.
        exclude_patterns: Glob-шаблоны, исключающие элементы из дерева.
        skip_empty_dirs: Прятать пустые каталоги при include-фильтрах.
        no_ignore: Отключить встроенные ignore-списки.
        extra_dir_ignores: Дополнительные имена каталогов для игнора.
        extra_file_ignores: Дополнительные имена/расширения файлов для игнора.
        include_json: Добавлять ли структурированное представление (``tree``) в ответ.

    Returns:
        dict: Сводка по каталогу, текстовое дерево и сериализованное представление.

    Example:
        >>> list_structure('.', max_depth=1, include_json=False)
    """
    return _list_structure_impl(
        root=root,
        max_depth=max_depth,
        include_files=include_files,
        include_hidden=include_hidden,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        skip_empty_dirs=skip_empty_dirs,
        no_ignore=no_ignore,
        extra_dir_ignores=extra_dir_ignores,
        extra_file_ignores=extra_file_ignores,
        include_json=include_json,
    )


def list_structure(*args, **kwargs) -> Dict:
    """Синхронная обёртка вокруг MCP-инструмента :func:`list_structure_tool`."""
    return list_structure_tool.fn(*args, **kwargs)


def _list_targets_impl(
    root: str,
    patterns: List[str],
    max_depth: Optional[int] = None,
    include_hidden: bool = False,
    skip_empty_dirs: bool = True,
    no_ignore: bool = False,
    extra_dir_ignores: Optional[List[str]] = None,
    extra_file_ignores: Optional[List[str]] = None,
    include_tree: bool = True,
) -> Dict:
    if not patterns:
        raise ValueError("patterns должен содержать хотя бы один глоб")

    filters = _build_filters(
        include_files=True,
        include_hidden=include_hidden,
        max_depth=max_depth,
        include_patterns=patterns,
        exclude_patterns=None,
        skip_empty_dirs=skip_empty_dirs,
        no_ignore=no_ignore,
        extra_dir_ignores=extra_dir_ignores,
        extra_file_ignores=extra_file_ignores,
    )
    node, stats = _scan_tree(root, filters)
    matches = _collect_files(node)

    response: Dict[str, object] = {
        "root": node.path,
        "patterns": patterns,
        "summary": stats.to_dict(),
        "matches": matches,
    }
    if include_tree:
        response["tree_text"] = render_tree(node).rstrip()
    return response

@mcp.tool(
    name="list_targets",
    description="Получить список файлов, удовлетворяющих glob-шаблонам.",
)
def list_targets_tool(
    root: str,
    patterns: List[str],
    max_depth: Optional[int] = None,
    include_hidden: bool = False,
    skip_empty_dirs: bool = True,
    no_ignore: bool = False,
    extra_dir_ignores: Optional[List[str]] = None,
    extra_file_ignores: Optional[List[str]] = None,
    include_tree: bool = True,
) -> Dict:
    """Выдать файлы и каталоги, подходящие под заданные glob-паттерны.

    Args:
        root: Каталог, внутри которого выполняется поиск.
        patterns: Список glob-шаблонов (``['*.py', 'src/**']`` и т.п.).
        max_depth: Максимальная глубина обхода.
        include_hidden: Включать ли скрытые файлы и каталоги.
        skip_empty_dirs: Прятать пустые каталоги при include-фильтрах.
        no_ignore: Отключить стандартные ignore-списки.
        extra_dir_ignores: Дополнительные каталоги для игнора.
        extra_file_ignores: Дополнительные файлы/расширения для игнора.
        include_tree: Добавлять ли текст дерева для совпавших директорий.

    Returns:
        dict: Список совпадений и статистика обхода.

    Example:
        >>> list_targets('.', patterns=['*.py'], max_depth=2)
    """
    return _list_targets_impl(
        root=root,
        patterns=patterns,
        max_depth=max_depth,
        include_hidden=include_hidden,
        skip_empty_dirs=skip_empty_dirs,
        no_ignore=no_ignore,
        extra_dir_ignores=extra_dir_ignores,
        extra_file_ignores=extra_file_ignores,
        include_tree=include_tree,
    )


def list_targets(*args, **kwargs) -> Dict:
    """Синхронная обёртка вокруг MCP-инструмента :func:`list_targets_tool`."""
    return list_targets_tool.fn(*args, **kwargs)

def _search_entries_impl(
    root: str,
    query: str,
    max_depth: Optional[int] = None,
    include_hidden: bool = False,
    include_files: bool = True,
    skip_empty_dirs: bool = False,
    no_ignore: bool = False,
    extra_dir_ignores: Optional[List[str]] = None,
    extra_file_ignores: Optional[List[str]] = None,
    limit: int = 50,
    include_tree: bool = True,
) -> Dict:
    if not query.strip():
        raise ValueError("query не должен быть пустым")

    filters = _build_filters(
        include_files=include_files,
        include_hidden=include_hidden,
        max_depth=max_depth,
        include_patterns=None,
        exclude_patterns=None,
        skip_empty_dirs=skip_empty_dirs,
        no_ignore=no_ignore,
        extra_dir_ignores=extra_dir_ignores,
        extra_file_ignores=extra_file_ignores,
    )
    limit_value = None if limit <= 0 else limit
    config = ScanConfig(root=str(Path(root).expanduser().resolve()), filters=filters)
    node, matches, truncated = search_directory(config, query, limit=limit_value)

    base = Path(node.path)
    results: List[Dict[str, object]] = []
    for match in matches:
        entry: Dict[str, object] = {
            "name": match.name,
            "path": match.path,
            "type": "directory" if match.is_dir else "file",
            "relative_path": _relative_path(base, Path(match.path)),
        }
        if match.is_dir and include_tree:
            entry["tree_text"] = render_tree(match).rstrip()
        results.append(entry)

    return {
        "root": node.path,
        "query": query,
        "count": len(results),
        "results": results,
        "truncated": truncated and limit_value is not None,
    }

@mcp.tool(
    name="search_entries",
    description="Поиск файлов и директорий по подстроке (без учёта регистра).",
)
def search_entries_tool(
    root: str,
    query: str,
    max_depth: Optional[int] = None,
    include_hidden: bool = False,
    include_files: bool = True,
    skip_empty_dirs: bool = False,
    no_ignore: bool = False,
    extra_dir_ignores: Optional[List[str]] = None,
    extra_file_ignores: Optional[List[str]] = None,
    limit: int = 50,
    include_tree: bool = True,
) -> Dict:
    """Найти элементы, чьё имя содержит указанную подстроку.

    Args:
        root: Каталог, из которого начинается поиск.
        query: Фрагмент имени (поиск без учёта регистра).
        max_depth: Глубина обхода, ``None`` — без ограничения.
        include_hidden: Включать ли скрытые файлы и директории.
        include_files: Включать ли файлы в поиск (выключите, чтобы искать только каталоги).
        skip_empty_dirs: Прятать пустые каталоги при include-фильтрах.
        no_ignore: Отключить стандартные ignore-списки.
        extra_dir_ignores: Дополнительные каталоги для игнора.
        extra_file_ignores: Дополнительные файлы/расширения для игнора.
        limit: Максимальное число совпадений (``<=0`` — без ограничения).
        include_tree: Добавлять ли текст дерева для совпавших директорий.

    Returns:
        dict: Найденные элементы с относительными путями и информацией о срезе.

    Example:
        >>> search_entries('.', 'settings', limit=10)
    """
    return _search_entries_impl(
        root=root,
        query=query,
        max_depth=max_depth,
        include_hidden=include_hidden,
        include_files=include_files,
        skip_empty_dirs=skip_empty_dirs,
        no_ignore=no_ignore,
        extra_dir_ignores=extra_dir_ignores,
        extra_file_ignores=extra_file_ignores,
        limit=limit,
        include_tree=include_tree,
    )


def search_entries(*args, **kwargs) -> Dict:
    """Синхронная обёртка вокруг MCP-инструмента :func:`search_entries_tool`."""
    return search_entries_tool.fn(*args, **kwargs)

def serve(host: str = "127.0.0.1", port: int = 8765, debug: bool = False) -> int:
    log_level = "debug" if debug else "info"
    try:
        asyncio.run(mcp.run_http_async(host=host, port=port, log_level=log_level))
    except KeyboardInterrupt:
        return 0
    return 0


def _scan_tree(root: str, filters: ScanFilters) -> Tuple[DirectoryNode, TreeStats]:
    path = Path(root).expanduser()
    if not path.exists():
        raise ValueError(f"Путь не найден: {path}")
    if not path.is_dir():
        raise ValueError(f"Это не директория: {path}")
    config = ScanConfig(root=str(path.resolve()), filters=filters)
    node = scan_tree(config)
    stats = _summarize(node)
    return node, stats


def _build_filters(
    *,
    include_files: bool,
    include_hidden: bool,
    max_depth: Optional[int],
    include_patterns: Optional[Sequence[str]],
    exclude_patterns: Optional[Sequence[str]],
    skip_empty_dirs: bool,
    no_ignore: bool,
    extra_dir_ignores: Optional[Sequence[str]],
    extra_file_ignores: Optional[Sequence[str]],
) -> ScanFilters:
    dir_ignores: List[str] = [] if no_ignore else list(DEFAULT_DIR_IGNORES)
    file_ignores: List[str] = [] if no_ignore else list(DEFAULT_FILE_IGNORES)
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
        skip_empty_dirs=skip_empty_dirs,
    )


def _normalize_patterns(patterns: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not patterns:
        return ()
    cleaned: List[str] = []
    for pattern in patterns:
        if not pattern:
            continue
        parts = [part.strip() for part in pattern.split(",")]
        cleaned.extend(part for part in parts if part)
    return tuple(dict.fromkeys(cleaned))


def _collect_files(node: DirectoryNode) -> List[str]:
    root_path = Path(node.path)
    results: List[str] = []

    def walk(current: DirectoryNode) -> None:
        if current.is_dir:
            for child in current.children:
                walk(child)
            return
        try:
            relative = Path(current.path).resolve().relative_to(root_path)
        except ValueError:
            relative = Path(current.path)
        results.append(str(relative))

    walk(node)
    results.sort()
    return results


def _summarize(node: DirectoryNode) -> TreeStats:
    directories = 0
    files = 0
    depth = 0

    def visit(current: DirectoryNode, level: int) -> None:
        nonlocal directories, files, depth
        depth = max(depth, level)
        if current.is_dir:
            directories += 1
            for child in current.children:
                visit(child, level + 1)
        else:
            files += 1

    visit(node, 0)
    return TreeStats(directories=directories, files=files, depth=depth)

def _relative_path(base: Path, target: Path) -> str:
    try:
        rel = target.resolve().relative_to(base.resolve())
        text = str(rel) if str(rel) else "."
    except Exception:
        text = str(target)
    return text.replace('\\', "/")



if __name__ == "__main__":
    mcp.run()
    #serve()
