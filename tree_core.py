"""Core directory scanning and serialization utilities shared across CLI, GUI, and MCP."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import os

DEFAULT_DIR_IGNORES: Tuple[str, ...] = (
    "Library",
    "Temp",
    "Logs",
    "Obj",
    "obj",
    "Build",
    "Builds",
    ".git",
    ".github",
    ".idea",
    ".vs",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".gradle",
)

DEFAULT_FILE_IGNORES: Tuple[str, ...] = (
    ".DS_Store",
    ".meta",
)

BRANCH = "|-- "
LAST_BRANCH = "`-- "
PIPE_PREFIX = "|   "
SPACE_PREFIX = "    "


@dataclass(frozen=True)
class ScanFilters:
    include_files: bool = True
    include_hidden: bool = False
    max_depth: Optional[int] = None
    dir_ignores: Tuple[str, ...] = DEFAULT_DIR_IGNORES
    file_ignores: Tuple[str, ...] = DEFAULT_FILE_IGNORES
    include_patterns: Tuple[str, ...] = ()
    exclude_patterns: Tuple[str, ...] = ()
    skip_empty_dirs: bool = False


@dataclass(frozen=True)
class ScanConfig:
    root: str
    filters: ScanFilters = ScanFilters()

    def normalized_root(self) -> Path:
        return Path(self.root).expanduser().resolve()


@dataclass
class DirectoryNode:
    name: str
    path: str
    is_dir: bool
    children: List["DirectoryNode"] = field(default_factory=list)
    size: Optional[int] = None
    mtime: Optional[float] = None

    def display_name(self) -> str:
        return f"{self.name}/" if self.is_dir else self.name

    def render_text(self) -> str:
        lines: List[str] = []

        def walk(node: "DirectoryNode", prefix: str, is_root: bool, is_last: bool) -> None:
            label = node.display_name()
            if is_root:
                lines.append(label)
            else:
                connector = LAST_BRANCH if is_last else BRANCH
                lines.append(prefix + connector + label)
            if not node.children:
                return
            next_prefix = prefix + (SPACE_PREFIX if is_last else PIPE_PREFIX)
            for index, child in enumerate(node.children):
                walk(child, next_prefix, False, index == len(node.children) - 1)

        walk(self, "", True, True)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "path": self.path,
            "is_dir": self.is_dir,
            "size": self.size,
            "mtime": self.mtime,
            "children": [child.to_dict() for child in self.children],
        }


class DirectoryScanner:
    """Filesystem walker, возвращающий дерево DirectoryNode по заданным фильтрам."""

    def __init__(self, filters: ScanFilters) -> None:
        self.filters = filters

    def scan(self, root: Path) -> DirectoryNode:
        if not root.exists():
            raise FileNotFoundError(f"Путь не найден: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Это не директория: {root}")
        stat_info = root.stat()
        node = DirectoryNode(
            name=root.name or str(root),
            path=str(root),
            is_dir=True,
            mtime=stat_info.st_mtime,
        )
        if self.filters.max_depth is not None and self.filters.max_depth <= 0:
            return node
        node.children = self._scan_directory(root, depth=1, rel_path="")
        return node

    def _scan_directory(self, path: Path, depth: int, rel_path: str) -> List[DirectoryNode]:
        if self.filters.max_depth is not None and depth > self.filters.max_depth:
            return []
        entries: List[DirectoryNode] = []
        try:
            raw_entries = list(os.scandir(path))
        except PermissionError:
            return entries
        raw_entries.sort(key=_sort_key)
        for entry in raw_entries:
            name = entry.name
            if not self.filters.include_hidden and name.startswith("."):
                continue
            entry_path = Path(entry.path)
            rel_child = _join_rel(rel_path, name)
            if entry.is_dir(follow_symlinks=False):
                if name in self.filters.dir_ignores:
                    continue
                if self._matches_pattern(rel_child, self.filters.exclude_patterns, True):
                    continue
                child_node = self._scan_directory_node(entry_path, depth, rel_child)
                if child_node is None:
                    continue
                entries.append(child_node)
            elif entry.is_file(follow_symlinks=False):
                if not self.filters.include_files:
                    continue
                if name in self.filters.file_ignores:
                    continue
                ext = entry_path.suffix
                if ext and ext in self.filters.file_ignores:
                    continue
                if self._matches_pattern(rel_child, self.filters.exclude_patterns, False):
                    continue
                if self.filters.include_patterns and not self._matches_pattern(
                    rel_child, self.filters.include_patterns, False
                ):
                    continue
                stat_info = entry.stat(follow_symlinks=False)
                entries.append(
                    DirectoryNode(
                        name=name,
                        path=str(entry_path),
                        is_dir=False,
                        size=stat_info.st_size,
                        mtime=stat_info.st_mtime,
                    )
                )
        return entries

    def _scan_directory_node(self, path: Path, depth: int, rel_path: str) -> Optional[DirectoryNode]:
        next_depth = depth + 1
        stat_info = path.stat()
        node = DirectoryNode(
            name=path.name or str(path),
            path=str(path),
            is_dir=True,
            mtime=stat_info.st_mtime,
        )
        if self.filters.max_depth is not None and depth >= self.filters.max_depth:
            node.children = []
        else:
            node.children = self._scan_directory(path, next_depth, rel_path)
        if self.filters.include_patterns:
            include_self = self._matches_pattern(rel_path, self.filters.include_patterns, True)
        else:
            include_self = True
        if self.filters.skip_empty_dirs and not node.children and not include_self:
            return None
        if not node.children and not include_self:
            return None
        return node

    @staticmethod
    def _matches_pattern(rel_path: str, patterns: Sequence[str], is_dir: bool) -> bool:
        if not patterns:
            return False
        target = rel_path.replace(os.sep, "/")
        if is_dir and not target.endswith("/"):
            target = f"{target}/"
        return any(fnmatch(target, pattern.rstrip()) for pattern in patterns)


def scan_tree(config: ScanConfig) -> DirectoryNode:
    scanner = DirectoryScanner(config.filters)
    return scanner.scan(config.normalized_root())


def render_tree(node: DirectoryNode) -> str:
    return node.render_text() + "\n"



def search_directory(config: ScanConfig, term: str, *, limit: Optional[int] = None) -> Tuple[DirectoryNode, List[DirectoryNode], bool]:
    """Отсканировать директорию и вернуть элементы, имя которых содержит term (без учёта регистра)."""
    term = term.strip()
    if not term:
        raise ValueError('Search term must be non-empty')
    root = scan_tree(config)
    matches, truncated = search_in_tree(root, term, limit=limit)
    return root, matches, truncated


def search_in_tree(root: DirectoryNode, term: str, *, limit: Optional[int] = None) -> Tuple[List[DirectoryNode], bool]:
    term_lower = term.lower()
    results: List[DirectoryNode] = []
    truncated = False

    def visit(node: DirectoryNode) -> None:
        nonlocal truncated
        if limit is not None and len(results) >= limit:
            truncated = True
            return
        if term_lower in node.name.lower():
            results.append(node)
            if limit is not None and len(results) >= limit:
                truncated = True
                return
        for child in node.children:
            visit(child)

    visit(root)
    return results, truncated


def _join_rel(parent: str, child: str) -> str:
    if not parent:
        return child
    return f"{parent}/{child}"


def _sort_key(entry: os.DirEntry) -> Tuple[int, str]:
    is_dir = entry.is_dir(follow_symlinks=False)
    return (0 if is_dir else 1, entry.name.lower())
