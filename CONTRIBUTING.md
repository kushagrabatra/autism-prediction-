# Contributing to Autism Detection System

Thank you for your interest in contributing! Here's how to get started.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/autism-prediction-.git`
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Create a new branch: `git checkout -b feature/your-feature-name`

## Development Workflow

1. Make your changes in the `ml_testing/` directory
2. Write or update tests in the `tests/` directory
3. Run tests before submitting: `pytest tests/`
4. Ensure your code follows PEP 8 style guidelines

## Submitting Changes

1. Commit your changes with a clear message: `git commit -m "feat: add new feature"`
2. Push to your fork: `git push origin feature/your-feature-name`
3. Open a Pull Request against the `main` branch

## Commit Message Convention

Use the following prefixes:
- `feat:` – new feature
- `fix:` – bug fix
- `docs:` – documentation update
- `test:` – test additions or updates
- `chore:` – maintenance tasks

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting Issues

Open an issue on GitHub with a clear description of the problem, steps to reproduce, and expected vs. actual behavior.
