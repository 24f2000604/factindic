"""Uploaded file → plain text."""
from __future__ import annotations

import io
import re
from pathlib import Path

TYPES = (".txt", ".md", ".pdf", ".docx")


class DocumentError(ValueError):
    pass


def extract(name: str, data: bytes) -> str:
    ext = Path(name or "").suffix.lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
            if not text.strip():
                raise DocumentError("This PDF has no text layer (a scan?). Paste the text instead.")
        elif ext == ".docx":
            from docx import Document
            text = "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
        elif ext in (".txt", ".md"):
            for enc in ("utf-8-sig", "utf-16") if data[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig",):
                try:
                    text = data.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise DocumentError("The text file is not UTF-8.")
        else:
            raise DocumentError(f"Unsupported file type {ext or '(none)'}. Use {', '.join(TYPES)}.")
    except DocumentError:
        raise
    except Exception as e:
        raise DocumentError(f"The file could not be read ({type(e).__name__}).") from e
    return re.sub(r"\s+", " ", text).strip()
