"""Language detection and per-language configuration."""

from __future__ import annotations

from pathlib import Path

# All languages supported by tree-sitter-languages
LANGUAGES: list[str] = [
    "python",
    "javascript",
    "typescript",
    "tsx",
    "jsx",
    "go",
    "rust",
    "java",
    "c",
    "cpp",
    "csharp",
    "ruby",
    "php",
    "swift",
    "kotlin",
    "scala",
    "haskell",
    "lua",
    "bash",
    "sql",
    "html",
    "css",
    "json",
    "yaml",
    "toml",
    "markdown",
    "dockerfile",
]

# Extension → language mapping (lowercase, no dot)
LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".sass": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".markdown": "markdown",
    "dockerfile": "dockerfile",
}

# tree-sitter node types that map to semantic chunks, per language
LANGUAGE_NODE_TYPES: dict[str, list[str]] = {
    "python": [
        "function_definition",
        "async_function_definition",
        "class_definition",
        "method_definition",
    ],
    "javascript": [
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "method_definition",
    ],
    "typescript": [
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
    ],
    "tsx": [
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
    ],
    "jsx": [
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "method_definition",
    ],
    "go": [
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "type_spec",
    ],
    "rust": [
        "function_item",
        "method_declaration",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "type_alias_item",
    ],
    "java": [
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "annotation_type_declaration",
    ],
    "c": [
        "function_definition",
        "struct_specifier",
        "enum_specifier",
        "typedefDeclaration",
    ],
    "cpp": [
        "function_definition",
        "method_declaration",
        "class_specifier",
        "struct_specifier",
        "enum_specifier",
        "namespace_definition",
    ],
    "csharp": [
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "struct_declaration",
        "enum_declaration",
        "delegate_declaration",
    ],
    "ruby": [
        "method",
        "class",
        "module",
        "block",
    ],
    "php": [
        "function_definition",
        "method",
        "class",
        "interface",
        "trait",
    ],
    "swift": [
        "function_declaration",
        "method_declaration",
        "class_declaration",
        "struct_declaration",
        "enum_declaration",
        "protocol_declaration",
    ],
    "kotlin": [
        "function_declaration",
        "method_declaration",
        "class_declaration",
        "object_declaration",
        "interface_declaration",
        "enum_entry",
    ],
    "scala": [
        "function_definition",
        "class_definition",
        "object_definition",
        "trait_definition",
        "interface_definition",
    ],
    "lua": [
        "function_declaration",
        "local_function",
        "assignment",
    ],
    "bash": [
        "function_definition",
        "command",
    ],
    "sql": [
        "select_statement",
        "create_table_statement",
        "create_index_statement",
        "view_definition",
    ],
    "html": [
        "element",
    ],
    "css": [
        "rule_set",
    ],
    "json": [
        "pair",
        "object",
        "array",
    ],
    "yaml": [
        "block_mapping_pair",
    ],
    "toml": [
        "table",
        "pair",
    ],
    "markdown": [
        "section",
        "fenced_code_block",
    ],
    "dockerfile": [
        "from_instruction",
        "run_instruction",
        "cmd_instruction",
        "label_instruction",
        "expose_instruction",
        "env_instruction",
        "add_instruction",
        "copy_instruction",
        "entrypoint_instruction",
        "workdir_instruction",
        "arg_instruction",
        "user_instruction",
        "volume_instruction",
        "instruction",
    ],
}

# Default glob patterns per language for file discovery
DEFAULT_GLOB_PATTERNS: dict[str, str] = {
    "python": "**/*.py",
    "javascript": "**/*.js",
    "typescript": "**/*.ts",
    "tsx": "**/*.tsx",
    "jsx": "**/*.jsx",
    "go": "**/*.go",
    "rust": "**/*.rs",
    "java": "**/*.java",
    "c": "**/*.c",
    "cpp": "**/*.cpp",
    "csharp": "**/*.cs",
    "ruby": "**/*.rb",
    "php": "**/*.php",
    "swift": "**/*.swift",
    "kotlin": "**/*.kt",
    "scala": "**/*.scala",
    "lua": "**/*.lua",
    "bash": "**/*.sh",
    "sql": "**/*.sql",
    "html": "**/*.html",
    "css": "**/*.css",
    "json": "**/*.json",
    "yaml": "**/*.yaml",
    "toml": "**/*.toml",
    "markdown": "**/*.md",
    "dockerfile": "**/Dockerfile",
}


def detect_language(file_path: str | Path) -> str | None:
    """
    Detect the programming language of a file based on its extension.

    Args:
        file_path: Path to the source file.

    Returns:
        Language string (e.g., "python") or None if unsupported.
    """
    path = Path(file_path)
    name = path.name.lower()

    # Special case for Dockerfile
    if name == "dockerfile" or name.startswith("dockerfile."):
        return "dockerfile"

    suffix = path.suffix.lower()
    return LANGUAGE_EXTENSIONS.get(suffix)


def is_supported_language(language: str) -> bool:
    """Check if a language is supported."""
    return language.lower() in LANGUAGES


def get_glob_patterns(
    languages: list[str] | None = None,
    custom: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Build a mapping of language → glob pattern for file discovery.

    Args:
        languages: List of languages to include (None = all supported).
        custom: Override patterns for specific languages.

    Returns:
        Dict mapping language → glob pattern string.
    """
    patterns = {}
    for lang in LANGUAGES:
        if languages and lang not in languages:
            continue
        patterns[lang] = (custom or {}).get(lang, DEFAULT_GLOB_PATTERNS.get(lang, f"**/*.{lang}"))
    return patterns


def get_node_types(language: str) -> list[str]:
    """Get the list of AST node types to extract for a language."""
    return LANGUAGE_NODE_TYPES.get(language.lower(), [])
