# LOFT (Logical Ontological Framework Translator)

## Project Overview

LOFT is a research project focusing on **Reflexive Neuro-Symbolic AI**. It bridges the gap between:
*   **Symbolic Core**: Explicit, logical reasoning using Answer Set Programming (ASP).
*   **Neural Interface**: Flexible pattern recognition and rule generation using Large Language Models (LLMs).

The system features a **self-reflexive core** that can question its logic, use LLMs to generate new rules, and validly incorporate them through a rigorous multi-stage pipeline.

## Tech Stack & Dependencies

*   **Language**: Python 3.11+
*   **Symbolic Reasoning**: `clingo` (ASP solver)
*   **LLM Interface**: `anthropic`, `openai`, `instructor` (for structured output)
*   **Data Validation**: `pydantic`
*   **Testing**: `pytest`, `hypothesis`
*   **Tooling**: `ruff` (lint/format), `mypy` (types)

## Setup & Configuration

1.  **Installation**:
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt # For dev tools
    ```

2.  **Environment Variables**:
    *   Copy `.env.example` to `.env`.
    *   Configure API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).

## Development Workflow

### Code Quality
Adhere to the project's strict validation and type safety standards.

*   **Linting**: `ruff check loft/`
*   **Formatting**: `ruff format loft/`
*   **Type Checking**: `mypy loft/` (Strict typing required)

### Testing
Testing is paramount given the self-modifying nature of the system.

*   **Run All Tests**: `pytest`
*   **Run with Coverage**: `pytest --cov=loft --cov-report=html`
*   **Specific Test**: `pytest tests/unit/legal/test_statute_of_frauds.py -v`

### Commit Standards
*   Start commit messages with a clear, concise summary.
*   Focus on *why* a change was made.
*   Ensure all tests pass before committing.

## Project Structure

*   `loft/`: Main package source code.
    *   `autonomous/`: Self-reflexive loops and agents.
    *   `core/`: Core definitions and abstractions.
    *   `legal/`: Domain-specific legal logic modules.
    *   `neural/`: LLM interaction and prompt management.
    *   `symbolic/`: ASP rule management and solver integration.
    *   `translation/`: NL <-> ASP translation layer.
    *   `validation/`: Multi-stage validation logic.
*   `asp_rules/`: Raw Answer Set Programming (.lp) files.
*   `datasets/`: Validation datasets (e.g., `adverse_possession`, `contracts`).
*   `examples/`: Usage demonstrations (Start here to understand the system).
*   `tests/`: Comprehensive test suite (Unit, Integration, E2E).
*   `CLAUDE.md`: Detailed architectural guidelines and principles.
*   `ROADMAP.md`: Project phases and MVP criteria.

## Key Architectural Concepts

1.  **Self-Reflexivity**: The system must be able to reason about its own reasoning. Code changes should preserve this capability.
2.  **Ontological Bridge**: Translation between Symbolic (ASP) and Natural Language must maintain semantic fidelity.
3.  **Validation**: Every generated rule undergoes syntactic, semantic, and empirical validation before being added to the core.
4.  **Stratification**: Logic is layered (Constitutional, Strategic, Tactical, Operational) to ensure safety and stability.
