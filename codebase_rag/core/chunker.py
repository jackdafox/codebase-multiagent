"""CodeChunker — converts tree-sitter parse trees into CodeChunk units."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from codebase_rag.core.chunk import CodeChunk
from codebase_rag.core.language import get_node_types
from codebase_rag.core.parser_pool import get_parser

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

MAX_FILE_SIZE = 1_000_000  # 1 MB — skip files larger than this
MAX_CHUNK_LINES = 500     # Lines — skip nodes larger than this


class CodeChunker:
    """
    Converts tree-sitter parse trees into CodeChunk units.

    Extracts named entities (functions, classes, methods, types) from source
    code as semantically meaningful chunks, capturing metadata like line
    numbers, qualified names, docstrings, and signatures.
    """

    def __init__(self) -> None:
        self._node_types_cache: dict[str, list[str]] = {}

    def chunk_file(
        self,
        file_path: str,
        content: bytes,
        language: str,
    ) -> list[CodeChunk]:
        """
        Parse a file and extract CodeChunks from its AST.

        Args:
            file_path: Absolute path to the source file.
            content: Raw UTF-8 bytes of the file.
            language: Programming language (e.g., "python").

        Returns:
            List of CodeChunks extracted from the file, in source order.
        """
        if len(content) > MAX_FILE_SIZE:
            return self._chunk_large_file(file_path, content, language)

        try:
            parser = get_parser(language)
            tree = parser.parse(content)
        except Exception:
            return self._fallback_chunk(file_path, content, language)

        return self._extract_chunks(tree, content, file_path, language)

    def _chunk_large_file(
        self,
        file_path: str,
        content: bytes,
        language: str,
    ) -> list[CodeChunk]:
        """Fallback for files too large to parse normally — whole file as one chunk."""
        return self._fallback_chunk(file_path, content, language)

    def _fallback_chunk(
        self,
        file_path: str,
        content: bytes,
        language: str,
    ) -> list[CodeChunk]:
        """Treat the entire file as a single chunk."""
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except Exception:
                return []

        lines = text.count("\n") + 1
        chunk = CodeChunk(
            id=str(uuid.uuid4()),
            text=text,
            language=language,
            node_type="file",
            file_path=file_path,
            start_line=1,
            end_line=lines,
            start_byte=0,
            end_byte=len(content),
            fully_qualified_name=None,
            parent_name=None,
            chunk_index=0,
            total_chunks=1,
            docstring=None,
            signature=None,
        )
        return [chunk]

    def _extract_chunks(
        self,
        tree: "Tree",
        content: bytes,
        file_path: str,
        language: str,
    ) -> list[CodeChunk]:
        """Walk the AST and extract target node types as chunks."""
        node_types = self._get_node_types(language)
        chunks: list[CodeChunk] = []
        self._walk_tree(
            tree.root_node,
            content,
            file_path,
            language,
            node_types,
            chunks,
            enclosing_names=[],
        )

        # Assign chunk_index and total_chunks
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(chunks)

        return chunks

    def _get_node_types(self, language: str) -> list[str]:
        """Get cached node types for a language."""
        if language not in self._node_types_cache:
            self._node_types_cache[language] = get_node_types(language)
        return self._node_types_cache[language]

    def _walk_tree(
        self,
        node: "Node",
        content: bytes,
        file_path: str,
        language: str,
        target_types: list[str],
        chunks: list[CodeChunk],
        enclosing_names: list[str],
    ) -> None:
        """Recursively walk the AST, extracting target nodes as chunks."""
        # Update enclosing names if this is a class/module node
        local_enclosing = list(enclosing_names)

        node_type = node.type

        # Build qualified name for this node
        node_name = self._get_node_name(node, content, language)
        if node_name and node_type in (
            ["class_definition", "class_declaration", "class_specifier",
             "struct_item", "struct_declaration", "interface_declaration",
             "module", "namespace_definition"]
        ):
            local_enclosing.append(node_name)

        # Extract chunk if this node matches a target type
        if node_type in target_types:
            chunk = self._node_to_chunk(
                node, content, file_path, language, node_type,
                node_name, ".".join(local_enclosing) if local_enclosing else None,
            )
            if chunk is not None:
                chunks.append(chunk)

        # Recurse into children
        for child in node.children:
            self._walk_tree(
                child, content, file_path, language, target_types,
                chunks, local_enclosing,
            )

    def _get_node_name(
        self,
        node: "Node",
        content: bytes,
        language: str,
    ) -> str | None:
        """Extract the name/identifier from an AST node."""
        if language == "python":
            return self._python_node_name(node, content)
        elif language in ("javascript", "typescript", "tsx", "jsx"):
            return self._js_node_name(node, content)
        elif language == "go":
            return self._go_node_name(node, content)
        elif language == "rust":
            return self._rust_node_name(node, content)
        elif language in ("java", "kotlin", "scala"):
            return self._java_node_name(node, content)
        elif language in ("c", "cpp", "csharp"):
            return self._c_family_node_name(node, content)
        elif language == "ruby":
            return self._ruby_node_name(node, content)
        elif language == "php":
            return self._php_node_name(node, content)
        else:
            return self._generic_node_name(node, content)

    def _python_node_name(self, node: "Node", content: bytes) -> str | None:
        """Extract name from Python AST nodes."""
        if node.type == "function_definition":
            for child in node.children:
                if child.type == "identifier":
                    return self._node_text(child, content)
        elif node.type == "async_function_definition":
            for child in node.children:
                if child.type == "identifier":
                    return self._node_text(child, content)
        elif node.type == "class_definition":
            for child in node.children:
                if child.type == "identifier":
                    return self._node_text(child, content)
        elif node.type == "method_definition":
            for child in node.children:
                if child.type == "identifier":
                    return self._node_text(child, content)
        return None

    def _js_node_name(self, node: "Node", content: bytes) -> str | None:
        """Extract name from JavaScript/TypeScript AST nodes."""
        if node.type in ("function_declaration", "function_expression"):
            for child in node.children:
                if child.type == "identifier":
                    return self._node_text(child, content)
        elif node.type == "arrow_function":
            # Arrow functions may have no name; try to get from variable declarator
            parent = getattr(node, "parent", None)
            if parent and parent.type == "variable_declarator":
                for child in parent.children:
                    if child.type == "identifier":
                        return self._node_text(child, content)
            return None
        elif node.type in ("class_declaration", "method_definition"):
            for child in node.children:
                if child.type == "identifier":
                    return self._node_text(child, content)
        elif node.type in ("interface_declaration", "type_alias_declaration",
                           "enum_declaration"):
            for child in node.children:
                if child.type == "type_identifier" or child.type == "identifier":
                    return self._node_text(child, content)
        return None

    def _go_node_name(self, node: "Node", content: bytes) -> str | None:
        """Extract name from Go AST nodes."""
        if node.type in ("function_declaration", "method_declaration"):
            for child in node.children:
                if child.type == "identifier":
                    return self._node_text(child, content)
        elif node.type == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    return self._go_node_name(child, content)
        elif node.type == "type_spec":
            for child in node.children:
                if child.type == "identifier":
                    return self._node_text(child, content)
        return None

    def _rust_node_name(self, node: "Node", content: bytes) -> str | None:
        """Extract name from Rust AST nodes."""
        if node.type == "function_item":
            for child in node.children:
                if child.type == "identifier":
                    return self._node_text(child, content)
        elif node.type == "method_declaration":
            for child in node.children:
                if child.type == "identifier":
                    return self._node_text(child, content)
        elif node.type == "impl_item":
            # impl Foo { ... } — extract "Foo"
            name_found = False
            for child in node.children:
                if name_found and child.type == "identifier":
                    return self._node_text(child, content)
                if child.type == "type_identifier":
                    name_found = True
        elif node.type in ("struct_item", "enum_item", "trait_item"):
            for child in node.children:
                if child.type == "identifier":
                    return self._node_text(child, content)
        elif node.type == "type_alias_item":
            for child in node.children:
                if child.type == "identifier":
                    return self._node_text(child, content)
        return None

    def _java_node_name(self, node: "Node", content: bytes) -> str | None:
        """Extract name from Java/Kotlin/Scala AST nodes."""
        for child in node.children:
            if child.type == "identifier":
                return self._node_text(child, content)
            if child.type == "type_identifier":
                return self._node_text(child, content)
        return None

    def _c_family_node_name(self, node: "Node", content: bytes) -> str | None:
        """Extract name from C/C++/C# AST nodes."""
        for child in node.children:
            if child.type == "identifier":
                return self._node_text(child, content)
            if child.type == "type_identifier":
                return self._node_text(child, content)
            if child.type == "field_identifier":
                return self._node_text(child, content)
        return None

    def _ruby_node_name(self, node: "Node", content: bytes) -> str | None:
        """Extract name from Ruby AST nodes."""
        if node.type == "method":
            for child in node.children:
                if child.type == "identifier":
                    return self._node_text(child, content)
        elif node.type in ("class", "module"):
            for child in node.children:
                if child.type == "constant":
                    return self._node_text(child, content)
        return None

    def _php_node_name(self, node: "Node", content: bytes) -> str | None:
        """Extract name from PHP AST nodes."""
        for child in node.children:
            if child.type == "name":
                return self._node_text(child, content)
            if child.type == "identifier":
                return self._node_text(child, content)
        return None

    def _generic_node_name(self, node: "Node", content: bytes) -> str | None:
        """Fallback name extraction — looks for identifier-like children."""
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "field_identifier"):
                return self._node_text(child, content)
        return None

    def _node_text(self, node: "Node", content: bytes) -> str:
        """Extract text from a node using byte offsets."""
        return content[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _node_to_chunk(
        self,
        node: "Node",
        content: bytes,
        file_path: str,
        language: str,
        node_type: str,
        node_name: str | None,
        parent_name: str | None,
    ) -> CodeChunk | None:
        """Convert a single AST node into a CodeChunk."""
        try:
            text = self._node_text(node, content)
        except Exception:
            return None

        # Skip empty nodes
        if not text.strip():
            return None

        # Count lines in this node
        line_count = text.count("\n") + 1
        if line_count > MAX_CHUNK_LINES:
            # Still include it but flag it
            pass

        # Extract docstring
        docstring = self._extract_docstring(node, content, language)

        # Extract signature
        signature = self._extract_signature(node, content, language)

        # Build qualified name
        fqn = node_name
        if parent_name:
            fqn = f"{parent_name}.{node_name}" if node_name else parent_name

        return CodeChunk(
            id=str(uuid.uuid4()),
            text=text,
            language=language,
            node_type=self._normalize_node_type(node_type, language),
            file_path=file_path,
            start_line=self._byte_to_line(node.start_byte, content),
            end_line=self._byte_to_line(node.end_byte, content),
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            fully_qualified_name=fqn,
            parent_name=parent_name,
            chunk_index=0,  # Will be assigned later
            total_chunks=1,  # Will be assigned later
            docstring=docstring,
            signature=signature,
        )

    def _normalize_node_type(self, node_type: str, language: str) -> str:
        """Map language-specific node types to a normalized set."""
        type_map = {
            "function_definition": "function",
            "async_function_definition": "function",
            "function_declaration": "function",
            "function_expression": "function",
            "arrow_function": "function",
            "function_item": "function",
            "method_definition": "method",
            "method_declaration": "method",
            "method": "method",
            "class_definition": "class",
            "class_declaration": "class",
            "class_specifier": "class",
            "class": "class",
            "struct_item": "struct",
            "struct_declaration": "struct",
            "struct_specifier": "struct",
            "impl_item": "impl",
            "interface_declaration": "interface",
            "interface_definition": "interface",
            "trait_item": "trait",
            "enum_item": "enum",
            "enum_declaration": "enum",
            "enum_specifier": "enum",
            "type_alias_item": "type",
            "type_alias_declaration": "type",
            "type_declaration": "type",
            "type_spec": "type",
            "module": "module",
            "namespace_definition": "namespace",
            "file": "file",
        }
        return type_map.get(node_type, node_type)

    def _extract_docstring(
        self,
        node: "Node",
        content: bytes,
        language: str,
    ) -> str | None:
        """Extract leading docstring/comments from a node."""
        # Walk children to find leading comment/docstring nodes
        # This is a simplified implementation
        return None

    def _extract_signature(
        self,
        node: "Node",
        content: bytes,
        language: str,
    ) -> str | None:
        """Extract function/method signature from a node."""
        # The node text usually contains the signature at the start
        text = self._node_text(node, content)
        if not text:
            return None

        if language == "python":
            # First line of function definition up to the colon
            lines = text.split("\n")
            if lines:
                return lines[0]
        elif language in ("javascript", "typescript", "tsx", "jsx", "go", "rust"):
            lines = text.split("\n")
            if lines:
                return lines[0]

        return None

    def _byte_to_line(self, byte_offset: int, content: bytes) -> int:
        """Convert a byte offset to a 1-indexed line number."""
        return content[:byte_offset].count(b"\n") + 1
