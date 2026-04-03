"""tree-sitter parser pool for efficient multi-language parsing."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

import tree_sitter_languages

if TYPE_CHECKING:
    from tree_sitter import Language, Parser

_POOL_LOCK = threading.Lock()
_PARSERS: dict[str, "Parser"] = {}


def get_parser(language: str) -> "Parser":
    """
    Get a tree-sitter parser for the given language.

    Parsers are cached thread-safely.

    Args:
        language: Language name (e.g., "python", "javascript").

    Returns:
        A configured tree-sitter Parser instance.
    """
    lang_lower = language.lower()

    # Quick check without lock
    if lang_lower in _PARSERS:
        return _PARSERS[lang_lower]

    with _POOL_LOCK:
        # Double-check after acquiring lock
        if lang_lower in _PARSERS:
            return _PARSERS[lang_lower]

        parser = tree_sitter_languages.get_parser(lang_lower)
        _PARSERS[lang_lower] = parser
        return parser


def get_language(language: str) -> "Language":
    """Get a tree-sitter Language object for the given language name."""
    return tree_sitter_languages.get_language(language.lower())


@contextmanager
def parser_for(language: str):
    """
    Context manager that yields a parser for the given language.

    Usage:
        with parser_for("python") as parser:
            tree = parser.parse(bytes(source, "utf-8"))
    """
    parser = get_parser(language)
    yield parser
