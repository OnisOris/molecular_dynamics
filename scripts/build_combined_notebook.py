"""Combine the six standalone laboratory notebooks into one Colab notebook."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

import nbformat

from embed_colab_sources import MODULE_ORDER, generated_cells, is_generated


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
OUTPUT = NOTEBOOK_DIR / "Молекулярная_динамика_единый_notebook.ipynb"
SOURCES = [
    NOTEBOOK_DIR / "00_Введение.ipynb",
    NOTEBOOK_DIR / "01_Периодические_границы.ipynb",
    NOTEBOOK_DIR / "02_Установление_равновесия.ipynb",
    NOTEBOOK_DIR / "03_Углеродные_стенки.ipynb",
    NOTEBOOK_DIR / "04_Фазовый_переход.ipynb",
    NOTEBOOK_DIR / "05_Неопределенность_и_чувствительность.ipynb",
]


def demote_headings(markdown: str) -> str:
    """Keep one H1 for the combined notebook and demote source headings."""
    return re.sub(r"^(#{1,5})(\s+)", r"#\1\2", markdown, flags=re.MULTILINE)


def source_cells(path: Path, prefix: str) -> list[nbformat.NotebookNode]:
    notebook = nbformat.read(path, as_version=4)
    cells: list[nbformat.NotebookNode] = []
    for index, original in enumerate(notebook.cells):
        if is_generated(original) or not original.source.strip():
            continue
        cell = deepcopy(original)
        if cell.cell_type == "markdown":
            cell.source = demote_headings(cell.source)
        elif cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
        cell.id = f"{prefix}-{index:02d}"
        tags = list(cell.metadata.get("tags", []))
        cell.metadata["tags"] = tags + [f"combined-{prefix}"]
        cells.append(cell)
    return cells


def main() -> None:
    first = nbformat.read(SOURCES[0], as_version=4)
    notebook = nbformat.v4.new_notebook(metadata=deepcopy(first.metadata))
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Молекулярная динамика: единый лабораторный notebook\n\n"
            "В документ последовательно объединены введение и пять лабораторных работ. "
            "Он полностью автономен: физическая модель, численные методы, эксперименты "
            "и анализ находятся в обычных Markdown- и Python-ячейках.\n\n"
            "**Порядок работы:** выполните **Runtime → Run all**. По умолчанию каждая "
            "работа использует `MODE = 'DEMO'`. Для полного эксперимента измените режим "
            "на `FULL` в конфигурационной ячейке соответствующего раздела и перезапустите документ.\n\n"
            "## Содержание\n\n"
            "1. Введение в Jupyter и молекулярную динамику\n"
            "2. Периодические границы\n"
            "3. Установление равновесия\n"
            "4. Углеродные стенки\n"
            "5. Фазовый переход\n"
            "6. Неопределённость и чувствительность",
            metadata={"tags": ["combined-title"]},
        )
    ]
    notebook.cells[0].id = "combined-title"

    core_intro = nbformat.v4.new_markdown_cell(
        "## Общее вычислительное ядро\n\n"
        "Этот код используется всеми лабораторными работами и поэтому приведён один раз. "
        "Блоки идут от физических констант к силам, интегратору, визуализации и методам анализа. "
        "Крупные реализации свёрнуты по умолчанию, но остаются обычными редактируемыми ячейками.",
        metadata={"tags": ["combined-core-intro"]},
    )
    core_intro.id = "combined-core-intro"
    notebook.cells.append(core_intro)

    all_modules = set(MODULE_ORDER)
    notebook.cells.extend(generated_cells(all_modules)[1:])

    for task_number, path in enumerate(SOURCES):
        notebook.cells.extend(source_cells(path, f"task{task_number}"))

    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    print(f"created {OUTPUT.relative_to(ROOT)} with {len(notebook.cells)} cells")


if __name__ == "__main__":
    main()
