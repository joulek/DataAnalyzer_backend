"""Découpage des documents Markdown en passages indexables."""

import re
from pathlib import Path

from app.services.rag.config import MAX_CHUNK_CHARS

HEADING_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def _title_of(text: str, fallback: str) -> str:
    """Première ligne « # Titre » du document, sinon le nom du fichier."""
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else fallback


def _split_long(section: str, max_chars: int) -> list[str]:
    """Coupe une section trop longue sur les paragraphes."""
    if len(section) <= max_chars:
        return [section]
    parts, current = [], ""
    for paragraph in section.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}".strip()
        if len(candidate) > max_chars and current:
            parts.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def chunk_document(path: Path, root: Path) -> list[dict]:
    """Renvoie une liste de passages {id, text, source, title, section}."""
    text = path.read_text(encoding="utf-8")
    source = path.relative_to(root).as_posix()
    doc_title = _title_of(text, path.stem)

    # Découpage aux titres de niveau 2 ; le texte avant le premier titre est gardé
    positions = [m.start() for m in HEADING_PATTERN.finditer(text)]
    bounds = [0] + positions + [len(text)]
    sections = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        block = text[start:end].strip()
        if block:
            sections.append(block)

    chunks = []
    for index, block in enumerate(sections):
        heading = HEADING_PATTERN.search(block)
        section_title = heading.group(1).strip() if heading else doc_title
        for part_index, part in enumerate(_split_long(block, MAX_CHUNK_CHARS)):
            # Chaque passage garde le titre du document : utile pour la recherche
            body = f"{doc_title}\n\n{part}"
            chunks.append(
                {
                    "id": f"{source}#{index}-{part_index}",
                    "text": body,
                    "source": source,
                    "title": doc_title,
                    "section": section_title,
                }
            )
    return chunks


def chunk_knowledge_base(root: Path) -> list[dict]:
    """Découpe tous les fichiers .md du dossier de connaissances."""
    chunks = []
    for path in sorted(root.rglob("*.md")):
        chunks.extend(chunk_document(path, root))
    return chunks