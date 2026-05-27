"""
Input validation and prompt injection prevention.
Every user-facing question must pass through sanitize_question() before
entering any LLM prompt.
"""

import re

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"you\s+are\s+now",
    r"system\s+prompt",
    r"<\|.*?\|>",
    r"\\n\\nHuman:",
    r"\\n\\nAssistant:",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def sanitize_question(text: str) -> str:
    """
    Validate and clean a user-submitted question.
    Raises ValueError with a safe error code on rejection — never echo
    the reason back to the user in production to avoid oracle attacks.
    """
    # Strip null bytes and non-printable control characters
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    text = text.strip()

    if len(text) < 3:
        raise ValueError("question_too_short")

    if len(text) > 1000:
        raise ValueError("question_too_long")

    for pattern in _COMPILED:
        if pattern.search(text):
            raise ValueError("question_rejected")

    return text


def wrap_user_content(question: str) -> str:
    """
    Wrap the sanitized question in delimiters for use inside LLM prompts.
    Prevents the model from treating user content as system instructions.
    """
    return f"<user_question>{question}</user_question>"
