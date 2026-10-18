"""Embed the MD package into every teaching notebook for standalone Colab use.

Run this script after changing ``src/`` to refresh the copies stored in the
notebooks.  The generated setup cell writes the embedded sources to a temporary
directory at runtime, so no repository checkout is required.
"""

from __future__ import annotations

from pathlib import Path
import re

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
SOURCE_DIR = ROOT / "src"
SETUP_TAG = "embedded-md-source"

COMMON_MODULES = {
    "__init__.py",
    "boundaries.py",
    "config.py",
    "constants.py",
    "forces.py",
    "initialization.py",
    "integrators.py",
    "observables.py",
    "potentials.py",
    "simulation.py",
    "walls.py",
}

NOTEBOOK_MODULES = {
    "00_Введение.ipynb": {"__init__.py", "constants.py"},
    "01_Периодические_границы.ipynb": COMMON_MODULES | {"visualization.py"},
    "02_Установление_равновесия.ipynb": COMMON_MODULES | {"equilibration.py"},
    "03_Углеродные_стенки.ipynb": COMMON_MODULES | {"visualization.py"},
    "04_Фазовый_переход.ipynb": COMMON_MODULES | {"phase_analysis.py"},
    "05_Неопределенность_и_чувствительность.ipynb": COMMON_MODULES | {"uq.py"},
}


def embedded_setup(module_names: set[str]) -> str:
    entries = []
    for name in sorted(module_names):
        source = (SOURCE_DIR / name).read_text(encoding="utf-8")
        if "'''" in source:
            raise ValueError(f"{name} contains an unsupported triple-single quote")
        entries.append(f"    {name!r}: r'''{source}''',")

    sources = "\n".join(entries)
    return f'''# Самодостаточный setup для Google Colab: код модели встроен в notebook.
# Репозиторий, Google Drive и загрузка локальных .py-файлов не требуются.
from pathlib import Path
import sys
import tempfile

_EMBEDDED_SOURCES = {{
{sources}
}}

_RUNTIME_ROOT = Path(tempfile.gettempdir()) / "molecular_dynamics_colab"
_PACKAGE_DIR = _RUNTIME_ROOT / "src"
_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
for _filename, _source in _EMBEDDED_SOURCES.items():
    (_PACKAGE_DIR / _filename).write_text(_source, encoding="utf-8")

# Гарантируем, что импорты ниже используют именно встроенную копию.
sys.path.insert(0, str(_RUNTIME_ROOT)) if str(_RUNTIME_ROOT) not in sys.path else None
for _module_name in [name for name in sys.modules if name == "src" or name.startswith("src.")]:
    del sys.modules[_module_name]

PROJECT_ROOT = Path.cwd()
print(f"Встроенное MD-ядро готово: {{len(_EMBEDDED_SOURCES)}} модулей; результаты → {{PROJECT_ROOT / 'output'}}")
'''


def remove_repository_bootstrap(source: str) -> str:
    lines = source.splitlines()
    cleaned: list[str] = []
    skipping_if = False
    for line in lines:
        stripped = line.strip()
        if stripped in {"from pathlib import Path", "import sys"}:
            continue
        if stripped.startswith("# Ищем каталог проекта"):
            continue
        if stripped.startswith("search_paths =") or stripped.startswith("project_root ="):
            continue
        if stripped.startswith("PROJECT_ROOT = next("):
            continue
        if stripped.startswith("if str(PROJECT_ROOT) not in sys.path"):
            skipping_if = stripped.endswith(":") and "sys.path.insert" not in stripped
            continue
        if stripped.startswith("if str(project_root) not in sys.path"):
            continue
        if skipping_if and stripped.startswith("sys.path.insert"):
            skipping_if = False
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip() + "\n"


def update_prose(source: str) -> str:
    source = source.replace(
        "; подробная инструкция приведена в `notebooks/README.md`",
        ". Каждый notebook содержит весь код и запускается в Colab самостоятельно",
    )
    source = source.replace(
        " Подробная инструкция — `notebooks/README.md`.",
        " Notebook содержит весь необходимый код и запускается в Colab без клонирования репозитория.",
    )
    source = re.sub(
        r"В ячейке `import` подключает NumPy, Matplotlib и стандартные инструменты путей;.*?`history` будет словарём массивов, а `frames` — массивом координат для анимации\.",
        "Первая setup-ячейка разворачивает встроенную копию MD-ядра во временный каталог Colab; внешние файлы и репозиторий не нужны. В этой ячейке `import` подключает NumPy, Matplotlib и функции встроенного пакета `src`. `default_config()` возвращает настройки системы. Число шагов регулирует длину траектории, `sample_interval` — частоту сохранения физических величин, `frame_interval` — частоту кадров. `history` будет словарём массивов, а `frames` — массивом координат для анимации.",
        source,
        flags=re.DOTALL,
    )
    source = source.replace(
        "а `Path` задаёт переносимый путь от корня репозитория.",
        "а `Path` задаёт переносимый путь внутри текущей среды Colab.",
    )
    return source


def update_notebook(path: Path, module_names: set[str]) -> None:
    notebook = nbformat.read(path, as_version=4)
    notebook.cells = [
        cell for cell in notebook.cells if SETUP_TAG not in cell.metadata.get("tags", [])
    ]

    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = remove_repository_bootstrap(cell.source)
        elif cell.cell_type == "markdown":
            cell.source = update_prose(cell.source)

    first_code = next(i for i, cell in enumerate(notebook.cells) if cell.cell_type == "code")
    setup = nbformat.v4.new_code_cell(
        embedded_setup(module_names),
        metadata={"tags": [SETUP_TAG]},
    )
    setup.id = SETUP_TAG
    notebook.cells.insert(first_code, setup)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main() -> None:
    for notebook_name, module_names in NOTEBOOK_MODULES.items():
        path = NOTEBOOK_DIR / notebook_name
        update_notebook(path, module_names)
        print(f"updated {path.relative_to(ROOT)} ({len(module_names)} embedded modules)")


if __name__ == "__main__":
    main()
