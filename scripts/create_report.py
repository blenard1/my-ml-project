from __future__ import annotations

import textwrap
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


PAGE_WIDTH = 8.27
PAGE_HEIGHT = 11.69
LEFT = 0.8
TOP = 10.85
LINE_HEIGHT = 0.24
CHARS_PER_LINE = 88
LINES_PER_PAGE = 42


def parse_markdown(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            lines.append("")
        elif line.startswith("# "):
            lines.append(line[2:].upper())
            lines.append("")
        elif line.startswith("## "):
            lines.append(line[3:])
            lines.append("")
        elif line.startswith("- "):
            wrapped = textwrap.wrap(line, width=CHARS_PER_LINE, subsequent_indent="  ")
            lines.extend(wrapped)
        else:
            wrapped = textwrap.wrap(line, width=CHARS_PER_LINE)
            lines.extend(wrapped if wrapped else [""])
    return lines


def chunk_pages(lines: list[str]) -> list[list[str]]:
    pages: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if len(current) >= LINES_PER_PAGE:
            pages.append(current)
            current = []
        current.append(line)
    if current:
        pages.append(current)
    return pages


def write_pdf(markdown_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages = chunk_pages(parse_markdown(markdown_path))
    with PdfPages(pdf_path) as pdf:
        for page_number, page_lines in enumerate(pages, start=1):
            fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
            fig.patch.set_facecolor("white")
            y = TOP
            for line in page_lines:
                if line.isupper() and len(line) > 3:
                    size = 15
                    weight = "bold"
                elif line and not line.startswith(" ") and len(line) < 70 and page_lines.index(line) < 5:
                    size = 12
                    weight = "bold"
                else:
                    size = 9.3
                    weight = "normal"
                fig.text(LEFT / PAGE_WIDTH, y / PAGE_HEIGHT, line, ha="left", va="top", fontsize=size, weight=weight)
                y -= LINE_HEIGHT
            fig.text(0.5, 0.035, str(page_number), ha="center", va="center", fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    markdown_path = root / "artifacts" / "final_report.md"
    pdf_path = root / "artifacts" / "final_report.pdf"
    write_pdf(markdown_path, pdf_path)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
