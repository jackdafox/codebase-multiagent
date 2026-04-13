"""Pytest fixtures for codebase_rag tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest



@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_python_file(temp_dir):
    """Create a sample Python file for testing."""
    content = '''"""Sample module for testing."""


def hello(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


class Greeter:
    """A class-based greeter."""

    def __init__(self, greeting: str = "Hello"):
        self.greeting = greeting

    def greet(self, name: str) -> str:
        """Greet with custom greeting."""
        return f"{self.greeting}, {name}!"
'''
    path = temp_dir / "sample.py"
    path.write_text(content)
    return path


@pytest.fixture
def sample_js_file(temp_dir):
    """Create a sample JavaScript file for testing."""
    content = '''/**
 * Sample module for testing
 */
function hello(name) {
    return `Hello, ${name}!`;
}

class Greeter {
    constructor(greeting = "Hello") {
        this.greeting = greeting;
    }

    greet(name) {
        return `${this.greeting}, ${name}!`;
    }
}

module.exports = { hello, Greeter };
'''
    path = temp_dir / "sample.js"
    path.write_text(content)
    return path


@pytest.fixture
def sample_go_file(temp_dir):
    """Create a sample Go file for testing."""
    content = '''package sample

import "fmt"

func Hello(name string) string {
    return fmt.Sprintf("Hello, %s!", name)
}

type Greeter struct {
    Greeting string
}

func (g *Greeter) Greet(name string) string {
    return fmt.Sprintf("%s, %s!", g.Greeting, name)
}
'''
    path = temp_dir / "sample.go"
    path.write_text(content)
    return path
