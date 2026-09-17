"""Plist stream framing for ``powermetrics --format plist``.

The tool writes one XML property list per sample and separates the documents
with a NUL byte. Output arrives in arbitrary chunks, so callers keep the
unparsed remainder and append the next chunk to it.
"""

import plistlib
from typing import Any

DOCUMENT_END = b"</plist>"


def extract_documents(buffer: bytes) -> tuple[list[dict[str, Any]], bytes]:
    """Return the complete documents in ``buffer`` and the unterminated tail."""
    documents: list[dict[str, Any]] = []
    while True:
        marker = buffer.find(DOCUMENT_END)
        if marker < 0:
            return documents, buffer
        slice_end = marker + len(DOCUMENT_END)
        chunk = buffer[:slice_end].lstrip(b"\x00 \t\r\n")
        buffer = buffer[slice_end:]
        try:
            document = plistlib.loads(chunk)
        except Exception:  # a malformed sample must not stop collection
            continue
        if isinstance(document, dict):
            documents.append(document)
