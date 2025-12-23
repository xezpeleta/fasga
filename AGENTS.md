# Agent Instructions for FASGA

This file contains instructions and preferences for AI coding agents working on this project.

## Python Environment Management

**Use Astral UV for all Python operations:**

- Use `uv` instead of `pip` for package installation
- Use `uv run` to execute Python commands and scripts
- Use `uv sync` to install dependencies from pyproject.toml
- Use `uv add` to add new dependencies

### Examples

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add package-name

# Run tests
uv run pytest

# Run the CLI
uv run fasga audio.mp3 text.txt -o output.srt

# Run Python scripts
uv run python script.py
```

## Development Workflow

1. Project setup: `uv sync` (not `pip install -e .`)
2. Add dependencies: `uv add package-name` (not `pip install`)
3. Run tests: `uv run pytest` (not `pytest`)
4. Run CLI: `uv run fasga` (not `fasga`)

## Project Preferences

- **Alignment Strategy**: Whisper-anchored (Strategy 2)
- **Text Format**: TXT only (for now)
- **Subtitle Format**: SRT only (for now)
- **Interface**: CLI-focused

## Code Style

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Type hints are encouraged but not required
- Comprehensive docstrings for public functions

