# Directory Tool Documentation
Этот проект предоставляет профессиональный инструмент для просмотра структуры файловых проектов: одна консольная утилита `make.py` объединяет экспорт дерева, удобный PyQt5‑интерфейс и MCP‑сервер для интеграции с агентами.
<img width="1094" height="752" alt="image" src="https://github.com/user-attachments/assets/4a6d2a6a-da80-4b00-8d1c-56a796184fa0" />
<img width="1092" height="747" alt="image" src="https://github.com/user-attachments/assets/2df74b4a-cac7-4dd8-979a-eb3cf7209128" />
<img width="1096" height="753" alt="image" src="https://github.com/user-attachments/assets/47e43228-d875-4358-b284-21c672e7071a" />

## Коротко
`make.py` — единый вход для трёх режимов: экспорт структуры в текст, просмотр через PyQt5 GUI и публикация через MCP сервер. Все режимы делят общий сканер (`tree_core.py`), поэтому фильтры, формат дерева и результаты поиска совпадают.

## CLI
### Экспорт (`python make.py export …`)
- **Назначение:** сформировать текстовое дерево и при желании показать его в консоли.
- **Ключевые опции:** `--max-depth`, `--dirs-only`, `--include-pattern/--exclude-pattern`, `--include-hidden`, `--skip-empty-dirs`, `--dir-ignore`, `--file-ignore`, `--no-ignore`.
- **Пример:** `python make.py export D:\Work\Game --max-depth 2 --include-pattern "*.cs" --skip-empty-dirs` → сохранит `PROJECT_STRUCTURE.txt` (имя меняется через `--name`).

### Поиск (`python make.py search …`)
- **Назначение:** искать файлы и каталоги по подстроке (без учёта регистра) с теми же фильтрами, что и экспорт.
- **Опции:** те же фильтры + `--limit` (0 = без ограничения) и `--show-tree` для вывода поддерева совпавших директорий.
- **Пример:** `python make.py search . src --include-pattern "*.py" --limit 20` выведет список попаданий с относительными путями.

## GUI (`python make.py gui …` или `python gui_app.py …`)
- **Назначение:** интерактивный обзор дерева, фильтрация, экспорт в `.txt` и быстрый поиск.
- **Что внутри:**
  - Светлые/тёмные контролы заменены на компактный тёмный стиль с акцентным синим.
  - Вверху поле пути с кнопкой «Выбрать…», сразу под фильтрами расположены кнопки «Назад», «Обновить», «Экспорт…» — они находятся прямо над деревом.
  - Блок фильтров (`Файлы`, `Скрытые`, `Пропускать пустые`, глубина, include/exclude-глобы).
  - В правой панели над списком результатов размещён блок поиска: поле, кнопки «Найти» и «Сбросить». Двойной клик по каталогу открывает его, по файлу — выделяет его в дереве.
  - Левая часть — дерево каталога, правая — список найденных элементов и их детальная информация/поддерево.
- **Запуск:** `python make.py gui D:\Repo --include-hidden --max-depth 3`. PyQt5 обязателен: `pip install pyqt5`.

## MCP (`python make.py mcp …` или прямой импорт `mcp_server.py`)
- **Назначение:** предоставить структуру, списки файлов и результаты поиска внешней модели по протоколу MCP.
- **Инструменты и параметры:**
  - `list_structure` — дерево каталога. Параметры: `root`, `max_depth`, `include_files`, `include_hidden`, `include_patterns`, `exclude_patterns`, `skip_empty_dirs`, `no_ignore`, `extra_dir_ignores`, `extra_file_ignores`, `include_json`. Пример: `list_structure('.', max_depth=2, include_json=False)`.
  - `list_targets` — выдаёт файлы по glob-шаблонам. Параметры: `root`, `patterns`, `max_depth`, `include_hidden`, `skip_empty_dirs`, `no_ignore`, `extra_dir_ignores`, `extra_file_ignores`, `include_tree`. Пример: `list_targets('src', patterns=['*.py', '*/tests/**'], include_tree=False)`.
  - `search_entries` — регистронезависимый поиск по именам. Параметры: `root`, `query`, `max_depth`, `include_hidden`, `include_files`, `skip_empty_dirs`, `no_ignore`, `extra_dir_ignores`, `extra_file_ignores`, `limit`, `include_tree`. Пример: `search_entries('.', 'config', limit=10)`.
- **Установка:**
  1. `pip install fastmcp` (если нужен только сервер, PyQt5 не обязателен).
  2. Запускаться из папки, где лежат `make.py` и `mcp_server.py`.
- **Старт сервера:**
  ```bash
  python make.py mcp --host 0.0.0.0 --port 8765 --debug
  ```
- **Подключение модели:**
  1. Добавьте MCP-эндпойнт `http://<host>:<port>` в клиенте/агенте.
  2. Разрешите использование инструментов `list_structure`, `list_targets`, `search_entries`.
  3. Передавайте абсолютный `root` и нужные фильтры (глубина, скрытые, include/exclude, `limit` и т.д.).
  4. Ответы содержат `tree_text`, при необходимости JSON (`tree`), относительные пути, а также флаг `truncated`, если поиск урезан лимитом.
- **Прямое использование в Python:**
  ```python
  from mcp_server import search_entries
  print(search_entries('.', 'node', limit=5))
  ```

---
*Экспорт, GUI и MCP используют одну реализацию обхода и поиска (`tree_core.py`), поэтому результаты остаются согласованными между режимами.*
