"""Build self-contained, readable Colab notebooks from the teaching sources.

Every required ``src`` module becomes an ordinary Python cell. Relative package
imports are removed because earlier cells define the same names in the notebook
namespace. Each code cell is preceded by a short scientific explanation.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
SOURCE_DIR = ROOT / "src"
GENERATED_TAG_PREFIX = "inline-md-"

MODULE_ORDER = [
    "constants.py", "config.py", "boundaries.py", "potentials.py",
    "initialization.py", "observables.py", "walls.py", "forces.py",
    "integrators.py", "simulation.py", "visualization.py",
    "equilibration.py", "phase_analysis.py", "uq.py",
]

COMMON_MODULES = {
    "constants.py", "config.py", "boundaries.py", "potentials.py",
    "initialization.py", "observables.py", "walls.py", "forces.py",
    "integrators.py", "simulation.py",
}

NOTEBOOK_MODULES = {
    "00_Введение.ipynb": {"constants.py"},
    "01_Периодические_границы.ipynb": COMMON_MODULES | {"visualization.py"},
    "02_Установление_равновесия.ipynb": COMMON_MODULES | {"equilibration.py"},
    "03_Углеродные_стенки.ipynb": COMMON_MODULES | {"visualization.py"},
    "04_Фазовый_переход.ipynb": COMMON_MODULES | {"phase_analysis.py"},
    "05_Неопределенность_и_чувствительность.ipynb": COMMON_MODULES | {"uq.py"},
}

MODULE_DOCS = {
    "constants.py": (
        "Физические константы и параметры взаимодействия",
        "Задаём единицы СИ, массу атома калия и параметры потенциалов K–K и K–C. "
        "Все энергии относятся к одной взаимодействующей паре, а не к молю.",
    ),
    "config.py": (
        "Геометрия и параметры расчёта",
        "Собираем размер области, шаг интегрирования, число частиц, граничные условия "
        "и диагностические допуски в одну проверяемую конфигурацию.",
    ),
    "boundaries.py": (
        "Граничные условия",
        "Реализуем периодическое заворачивание координат, minimum-image расстояние "
        "и отражение от непериодических границ.",
    ),
    "potentials.py": (
        "Парный потенциал Леннарда — Джонса",
        "Вычисляем энергию и производную LJ, затем обрезаем и сдвигаем энергию на "
        "радиусе cutoff. Проверка отдельно останавливает численно опасные сближения.",
    ),
    "initialization.py": (
        "Начальные координаты и скорости",
        "Частицы размещаются на слегка возмущённой решётке. Скорости выбираются из "
        "распределения Максвелла, затем удаляется движение центра масс и задаётся температура.",
    ),
    "observables.py": (
        "Измеряемые физические величины",
        "Определяем кинетическую энергию, мгновенную температуру, двумерное вириальное "
        "давление и механическое давление на x-стенки.",
    ),
    "walls.py": (
        "Геометрия углеродных стенок",
        "Строим фиксированные ряды атомов углерода с шагом алмазной решётки. Эти атомы "
        "создают внешнее поле, но сами не интегрируются во времени.",
    ),
    "forces.py": (
        "Расчёт межатомных сил",
        "Суммируем парные силы, потенциальную энергию и вириал. Для больших систем "
        "используется список пространственных ячеек; поддерживается и взаимодействие со стенками.",
    ),
    "integrators.py": (
        "Интегрирование уравнений движения",
        "Один шаг velocity Verlet последовательно обновляет координаты, силы и скорости. "
        "Есть варианты для периодической системы и системы с фиксированными стенками.",
    ),
    "simulation.py": (
        "Движок молекулярной динамики",
        "Объединяем инициализацию, силы, интегратор и наблюдаемые величины. `MDEngine` "
        "выполняет шаги, а `run_nve` сохраняет временные ряды и кадры траектории.",
    ),
    "visualization.py": (
        "Графики и анимации",
        "Функции Matplotlib строят диагностические графики и сохраняют GIF-анимации. "
        "Этот блок не участвует в вычислении сил.",
    ),
    "equilibration.py": (
        "Критерий установления равновесия",
        "Временной ряд делится на блоки. Сравнение средних соседних участков даёт "
        "воспроизводимый индекс burn-in и статистику после него.",
    ),
    "phase_analysis.py": (
        "Кластерные признаки фазового перехода",
        "Частицы связываются по пороговому расстоянию с учётом границ. Размер крупнейшего "
        "кластера и его восприимчивость служат индикаторами конечносистемного перехода.",
    ),
    "uq.py": (
        "Неопределённость и суррогатная модель",
        "Для точки DOE запускаем MD и оцениваем давление, затем подгоняем линейную "
        "модель P(T,N) и переносим диапазоны входов методом Monte Carlo.",
    ),
}


def without_package_imports(source: str) -> str:
    """Remove relative imports while retaining original formatting and comments."""
    tree = ast.parse(source)
    removed_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            removed_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return "\n".join(
        line for number, line in enumerate(source.splitlines(), start=1)
        if number not in removed_lines
    ).strip() + "\n"


def is_generated(cell: nbformat.NotebookNode) -> bool:
    tags = cell.metadata.get("tags", [])
    return "embedded-md-source" in tags or any(
        tag.startswith(GENERATED_TAG_PREFIX) for tag in tags
    )


def clean_existing_code(source: str) -> str:
    """Remove repository bootstraps and imports now supplied by inline cells."""
    lines = source.splitlines()
    cleaned: list[str] = []
    skipping_if = False
    for line in lines:
        stripped = line.strip()
        if stripped in {"from pathlib import Path", "import sys", "import tempfile"}:
            continue
        if stripped.startswith("# Ищем каталог проекта"):
            continue
        if stripped.startswith(("search_paths =", "project_root =", "PROJECT_ROOT = next(")):
            continue
        if stripped.startswith("if str(PROJECT_ROOT) not in sys.path"):
            skipping_if = stripped.endswith(":") and "sys.path.insert" not in stripped
            continue
        if stripped.startswith("if str(project_root) not in sys.path"):
            continue
        if skipping_if and stripped.startswith("sys.path.insert"):
            skipping_if = False
            continue
        if re.match(r"from src(?:\.|\s)", stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip() + "\n"


def update_prose(source: str) -> str:
    source = source.replace(
        "Первая setup-ячейка разворачивает встроенную копию MD-ядра во временный каталог Colab; внешние файлы и репозиторий не нужны. В этой ячейке `import` подключает NumPy, Matplotlib и функции встроенного пакета `src`.",
        "Следующие Python-ячейки содержат полное MD-ядро непосредственно в notebook: физические параметры, силы, интегратор и функции анализа. Внешние `.py`-файлы и репозиторий не нужны. После них конфигурационная ячейка подключает NumPy и Matplotlib.",
    )
    source = source.replace("функции ядра `src`", "функции из предыдущих ячеек")
    source = source.replace(
        "Параметры K–K взяты из `src/constants.py`",
        "Параметры K–K заданы выше, в ячейке физических констант",
    )
    return source


def generated_cells(module_names: set[str]) -> list[nbformat.NotebookNode]:
    cells: list[nbformat.NotebookNode] = []
    intro = nbformat.v4.new_markdown_cell(
        "### Вычислительное ядро внутри notebook\n\n"
        "Ниже приведён весь код, необходимый для расчёта. Блоки расположены в порядке "
        "зависимостей: от физических параметров к силам и интегратору. Каждый блок можно "
        "читать, изменять и выполнять как обычную Python-ячейку; скрытой загрузки кода нет.",
        metadata={"tags": ["inline-md-intro"]},
    )
    intro.id = "inline-md-intro"
    cells.append(intro)

    environment = nbformat.v4.new_code_cell(
        "from pathlib import Path\n\n"
        "# В Colab это /content; локально — каталог, из которого открыт notebook.\n"
        "PROJECT_ROOT = Path.cwd()\n"
        "print(f'Рабочий каталог: {PROJECT_ROOT}; результаты будут сохранены в output/')",
        metadata={"tags": ["inline-md-environment"]},
    )
    environment.id = "inline-md-environment"
    cells.append(environment)

    for name in MODULE_ORDER:
        if name not in module_names:
            continue
        slug = name.removesuffix(".py").replace("_", "-")
        title, description = MODULE_DOCS[name]
        markdown = nbformat.v4.new_markdown_cell(
            f"#### {title}\n\n{description}",
            metadata={"tags": [f"inline-md-doc-{slug}"]},
        )
        markdown.id = f"inline-md-doc-{slug}"
        source = without_package_imports((SOURCE_DIR / name).read_text(encoding="utf-8"))
        code_metadata: dict[str, object] = {"tags": [f"inline-md-code-{slug}"]}
        # Colab and Jupyter/VS Code use different metadata keys for the same
        # affordance. Large implementation cells start folded but remain one
        # click away for reading and editing.
        if len(source.splitlines()) >= 80:
            code_metadata.update(
                {
                    "cellView": "form",
                    "collapsed": True,
                    "jupyter": {"source_hidden": True},
                }
            )
        code = nbformat.v4.new_code_cell(
            source,
            metadata=code_metadata,
        )
        code.id = f"inline-md-code-{slug}"
        cells.extend([markdown, code])
    return cells


def update_notebook(path: Path, module_names: set[str]) -> None:
    notebook = nbformat.read(path, as_version=4)
    notebook.cells = [cell for cell in notebook.cells if not is_generated(cell)]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = clean_existing_code(cell.source)
        elif cell.cell_type == "markdown":
            cell.source = update_prose(cell.source)

    first_code = next(i for i, cell in enumerate(notebook.cells) if cell.cell_type == "code")
    notebook.cells[first_code:first_code] = generated_cells(module_names)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main() -> None:
    for notebook_name, module_names in NOTEBOOK_MODULES.items():
        path = NOTEBOOK_DIR / notebook_name
        update_notebook(path, module_names)
        print(f"updated {path.relative_to(ROOT)} ({len(module_names)} inline code blocks)")


if __name__ == "__main__":
    main()
