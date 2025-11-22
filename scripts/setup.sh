#!/bin/bash
# LOFT Development Environment Setup Script
# This script sets up the complete development environment for LOFT

set -e  # Exit on error

echo "========================================="
echo "LOFT - Reflexive Neuro-Symbolic AI"
echo "Development Environment Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v python3.11 &> /dev/null; then
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python 3.11+ is required but not found"
        echo "Please install Python 3.11 or higher"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    REQUIRED_VERSION="3.11"

    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        echo "Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
        exit 1
    fi
    PYTHON=python3
else
    PYTHON=python3.11
fi

echo "✓ Using Python: $($PYTHON --version)"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "  Virtual environment already exists"
    read -p "  Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        $PYTHON -m venv .venv
        echo "  ✓ Recreated virtual environment"
    else
        echo "  Using existing virtual environment"
    fi
else
    $PYTHON -m venv .venv
    echo "✓ Created virtual environment"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded to $(pip --version | cut -d' ' -f2)"
echo ""

# Install package in development mode
echo "Installing LOFT package with dependencies..."
echo "  This may take a few minutes..."
pip install -e ".[dev]" --quiet

if [ $? -eq 0 ]; then
    echo "✓ Package installed successfully"
else
    echo "✗ Package installation failed"
    exit 1
fi
echo ""

# Verify Clingo installation
echo "Verifying Clingo (ASP solver) installation..."
python -c "import clingo; print(f'  ✓ Clingo version: {clingo.__version__}')"
if [ $? -ne 0 ]; then
    echo "  ✗ Clingo import failed"
    exit 1
fi
echo ""

# Verify other key dependencies
echo "Verifying key dependencies..."
python -c "import anthropic; print('  ✓ Anthropic SDK installed')" || echo "  ✗ Anthropic SDK failed"
python -c "import pydantic; print('  ✓ Pydantic installed')" || echo "  ✗ Pydantic failed"
python -c "import pytest; print('  ✓ Pytest installed')" || echo "  ✗ Pytest failed"
python -c "import loguru; print('  ✓ Loguru installed')" || echo "  ✗ Loguru failed"
echo ""

# Create .env from example if it doesn't exist
echo "Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env file from .env.example"
    echo ""
    echo "  ⚠️  IMPORTANT: Please edit .env and add your API keys:"
    echo "     - ANTHROPIC_API_KEY (get from https://console.anthropic.com/)"
    echo "     - OPENAI_API_KEY (optional, from https://platform.openai.com/)"
    echo ""
else
    echo "  .env file already exists (not overwriting)"
    echo ""
fi

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# Environment variables
.env

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.hypothesis/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Type checking
.mypy_cache/
.dmypy.json
dmypy.json
EOF
    echo "✓ Created .gitignore"
else
    echo "  .gitignore already exists"
fi
echo ""

# Run a simple test to verify everything works
echo "Running verification test..."
python -c "from loft.config import config; print(f'  ✓ Configuration loaded successfully')"
python -c "from loft.config import config; print(f'  ✓ LLM Provider: {config.llm.provider}')"
python -c "from loft.config import config; print(f'  ✓ ASP Programs Directory: {config.asp.programs_dir}')"
echo ""

# Success!
echo "========================================="
echo "✓ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Add your API keys to .env:"
echo "   nano .env  # or your preferred editor"
echo ""
echo "3. Verify installation:"
echo "   python -c 'from loft.config import config; print(config)'"
echo ""
echo "4. Run tests (once implemented):"
echo "   pytest"
echo ""
echo "5. Start developing!"
echo "   See ROADMAP.md for Phase 0 tasks"
echo ""
echo "For more information, see:"
echo "  - README.md: Project overview"
echo "  - ROADMAP.md: Development phases"
echo "  - CLAUDE.md: Development guidelines"
echo ""
