# Contributing to Enso Atlas

Thank you for your interest in contributing to Enso Atlas! This document provides guidelines for contributing to the project.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributions from everyone.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/med-gemma-hackathon.git
   cd med-gemma-hackathon
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Backend (Python)

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
ruff check src/
black src/ --check

# Run type checking
mypy src/

# Run tests
pytest tests/
```

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev     # Development server
npm run lint    # ESLint
npm run build   # Production build
```

## Submitting Changes

1. **Ensure your code follows the project style**:
   - Python: Use `black` for formatting, `ruff` for linting
   - TypeScript: Use `eslint` and `prettier`

2. **Write meaningful commit messages**:
   - Use present tense ("Add feature" not "Added feature")
   - Be descriptive but concise

3. **Push to your fork** and submit a Pull Request

4. **Describe your changes** in the PR description:
   - What problem does it solve?
   - How did you test it?
   - Any breaking changes?

## Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Update documentation if needed
- Add tests for new functionality
- Ensure all tests pass before submitting

## Reporting Issues

When reporting bugs, please include:

- Python/Node.js version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant error messages or logs

## Questions?

Feel free to open an issue for discussion or reach out to the maintainers.

---

Thank you for contributing!
