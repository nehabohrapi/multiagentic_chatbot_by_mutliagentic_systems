import re


def clean_document_text(text: str) -> str:
    """Normalize whitespace without removing numeric or scientific symbols."""
    text = text.replace("\x00", "")
    # Keep symbols such as °C, ≥, ±, ≤, %, decimal points, and table pipes.
    # Only horizontal whitespace is normalized; content is never token-filtered.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
