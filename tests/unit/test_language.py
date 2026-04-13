"""Unit tests for language detection."""

from __future__ import annotations

from codebase_rag.core.language import (
    detect_language,
    get_glob_patterns,
    get_node_types,
    is_supported_language,
    LANGUAGES,
)


class TestDetectLanguage:
    def test_python(self, tmp_path):
        assert detect_language("foo.py") == "python"
        assert detect_language("foo/bar.py") == "python"

    def test_javascript(self, tmp_path):
        assert detect_language("foo.js") == "javascript"
        assert detect_language("foo.ts") == "typescript"
        assert detect_language("foo.jsx") == "jsx"
        assert detect_language("foo.tsx") == "tsx"

    def test_go(self, tmp_path):
        assert detect_language("foo.go") == "go"

    def test_rust(self, tmp_path):
        assert detect_language("foo.rs") == "rust"

    def test_dockerfile(self, tmp_path):
        assert detect_language("Dockerfile") == "dockerfile"
        assert detect_language("dockerfile.prod") == "dockerfile"

    def test_unsupported_extension(self, tmp_path):
        assert detect_language("foo.xyz") is None

    def test_case_insensitive(self, tmp_path):
        assert detect_language("foo.PY") == "python"
        assert detect_language("foo.Js") == "javascript"


class TestIsSupportedLanguage:
    def test_supported(self):
        assert is_supported_language("python")
        assert is_supported_language("javascript")
        assert is_supported_language("go")
        assert is_supported_language("rust")

    def test_unsupported(self):
        assert not is_supported_language("cobol")
        assert not is_supported_language("fortran")


class TestGetGlobPatterns:
    def test_all_languages(self):
        patterns = get_glob_patterns()
        for lang in LANGUAGES:
            assert lang in patterns, f"Missing pattern for {lang}"

    def test_filtered_languages(self):
        patterns = get_glob_patterns(languages=["python", "javascript"])
        assert "python" in patterns
        assert "javascript" in patterns
        assert "go" not in patterns

    def test_custom_patterns(self):
        patterns = get_glob_patterns(
            languages=["python"],
            custom={"python": "src/**/*.py"},
        )
        assert patterns["python"] == "src/**/*.py"


class TestGetNodeTypes:
    def test_python_types(self):
        types = get_node_types("python")
        assert "function_definition" in types
        assert "class_definition" in types

    def test_javascript_types(self):
        types = get_node_types("javascript")
        assert "function_declaration" in types
        assert "class_declaration" in types

    def test_unknown_language(self):
        types = get_node_types("cobol")
        assert types == []
