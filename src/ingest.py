import re
from pathlib import Path

from schema import Chunk

DATA_DIR = Path(__file__).parent.parent / "data"

# Matches headers like: "# FAQ-01 — How do I access a course after purchasing it?"
HEADER_RE = re.compile(r"^#\s*([A-Z]+-\d+)\s*[—-]\s*(.+)$", re.MULTILINE)

SUPERSEDED_MARKERS = [
    "outdated", "no longer", "older", "previous", "obsolete", "archived",
    "used to", "is no longer", "not.*current", "retired",
]
SUPERSEDED_RE = re.compile("|".join(SUPERSEDED_MARKERS), re.IGNORECASE)


def _split_entries(raw_text: str):
    """Split a source file into (entry_id, title, body) tuples."""
    matches = list(HEADER_RE.finditer(raw_text))
    entries = []
    for i, m in enumerate(matches):
        entry_id, title = m.group(1), m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end]
        # cut off trailing section markers / horizontal rules
        body = re.split(r"\n-{3,}\n", body)[0]
        body = re.sub(r"^SECTION \d+.*$", "", body, flags=re.MULTILINE)
        body = body.strip()
        entries.append((entry_id, title, body))
    return entries


def _extract_last_reviewed(body: str):
    m = re.search(
        r"(Last reviewed|Last updated|Effective date|Effective|Updated|Reviewed):\s*([A-Za-z]+\s+\d{4}|\d{4})",
        body,
    )
    return m.group(2) if m else None


def _extract_status(body: str):
    m = re.search(r"^STATUS:\s*(.+)$", body, re.MULTILINE)
    return m.group(1).strip().rstrip(".") if m else None


def load_faqs():
    text = (DATA_DIR / "faqs.md").read_text(encoding="utf-8")
    chunks = []
    for entry_id, title, body in _split_entries(text):
        cid = entry_id.lower()  # "faq-01"
        chunks.append(Chunk(
            id=cid,
            source_type="faq",
            title=title,
            text=f"FAQ: {title}\n\n{body}",
            contains_superseded_note=bool(SUPERSEDED_RE.search(body)),
            tags=["faq"],
        ))
    return chunks


def load_policies():
    text = (DATA_DIR / "policies.md").read_text(encoding="utf-8")
    chunks = []
    for entry_id, title, body in _split_entries(text):
        cid = entry_id.lower()
        chunks.append(Chunk(
            id=cid,
            source_type="policy",
            title=title,
            text=f"POLICY: {title}\n\n{body}",
            last_reviewed=_extract_last_reviewed(body),
            contains_superseded_note=bool(SUPERSEDED_RE.search(body)),
            tags=["policy"],
        ))
    return chunks


def load_tickets():
    text = (DATA_DIR / "tickets.md").read_text(encoding="utf-8")
    chunks = []
    for entry_id, title, body in _split_entries(text):
        cid = entry_id.lower()
        status = _extract_status(body)
        chunks.append(Chunk(
            id=cid,
            source_type="ticket",
            title=title,
            text=f"PAST SUPPORT TICKET: {title}\n\n{body}",
            status=status,
            contains_superseded_note=bool(SUPERSEDED_RE.search(body)),
            tags=["ticket"],
        ))
    return chunks


def load_all_chunks():
    return load_faqs() + load_policies() + load_tickets()


if __name__ == "__main__":
    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks:")
    by_type = {}
    for c in chunks:
        by_type.setdefault(c.source_type, 0)
        by_type[c.source_type] += 1
    print(by_type)
    flagged = [c.id for c in chunks if c.contains_superseded_note]
    print(f"{len(flagged)} chunks flagged as containing superseded/outdated info: {flagged}")
