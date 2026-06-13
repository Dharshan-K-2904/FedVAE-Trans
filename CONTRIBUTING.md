# Contributing to FedVAE-Trans

Thank you for your interest in contributing to **FedVAE-Trans**! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

- Be respectful and inclusive in all interactions
- Provide constructive feedback
- Focus on ideas, not individuals
- Help maintain a positive, collaborative environment

## Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/PES-University/FedVAE-Trans.git
cd FedVAE-Trans
git remote add upstream https://github.com/PES-University/FedVAE-Trans.git
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### 3. Set Up Development Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"  # Install with development dependencies
```

## Development Workflow

### Code Style

Follow **PEP 8** conventions:

```bash
# Format code with Black
black src/ --line-length 100

# Check style with Flake8
flake8 src/ --max-line-length=100

# Sort imports with isort
isort src/
```

### Type Hints

All functions must include type hints:

```python
def train_model(
    model: BiLSTMTransVAE,
    data_loader: DataLoader,
    epochs: int,
    device: str = "cuda"
) -> Dict[str, float]:
    """Train the model and return metrics."""
    ...
```

### Documentation

Document all public methods with docstrings:

```python
def detect_anomaly(
    self,
    log_sequence: torch.Tensor,
    threshold_percentile: float = 95
) -> Tuple[float, bool]:
    """
    Detect anomalies in log sequences using reconstruction error.
    
    Args:
        log_sequence: Input tensor of shape (seq_len, embedding_dim)
        threshold_percentile: Anomaly threshold percentile (0-100)
    
    Returns:
        Tuple of (reconstruction_error, is_anomaly)
    
    Raises:
        ValueError: If threshold_percentile not in [0, 100]
    """
    ...
```

### Testing

Write tests for all new features:

```bash
# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=html
```

Test file naming: `test_module_name.py`

```python
# Example test
import pytest
from src.models.bilstm_trans_vae import BiLSTMTransVAE

def test_model_initialization():
    """Test that model initializes correctly."""
    model = BiLSTMTransVAE(input_dim=128, hidden_dim=128)
    assert model is not None
    
def test_forward_pass():
    """Test forward pass with valid input."""
    model = BiLSTMTransVAE(input_dim=128, hidden_dim=128)
    x = torch.randn(32, 100, 128)  # batch_size=32, seq_len=100
    output = model(x)
    assert output["reconstruction"].shape == x.shape
```

## Commit Guidelines

### Message Format

```
<type>: <subject>

<body>

<footer>
```

### Types
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
- `perf:` Performance improvement
- `style:` Code style
- `ci:` CI/CD

### Examples

```
feat: Add FedProx regularization to aggregation strategy

Implements FedProx proximal regularization to handle Non-IID data
heterogeneity in federated learning. Adds proximal_mu parameter
for penalty weight control.

Fixes #42
```

```
fix: Resolve CUDA out-of-memory during training

Reduce default batch size from 64 to 32 and add gradient accumulation
to prevent OOM errors on GPUs with <8GB memory.

Closes #38
```

## Pull Request Process

### 1. Before Submitting

- [ ] Tests pass: `pytest tests/ -v`
- [ ] Code is formatted: `black src/ && isort src/`
- [ ] No style issues: `flake8 src/`
- [ ] Documentation is updated
- [ ] Changelog is updated (if applicable)

### 2. PR Title

Follow the same format as commits:
```
feat: Add federated learning with FedProx aggregation
fix: Resolve gradient explosion in VAE training
```

### 3. PR Description

```markdown
## Description
Brief explanation of changes

## Related Issue
Fixes #123 (or References #456)

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing approach:
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No new warnings generated
```

### 4. Review Process

- Project maintainers will review your PR
- Address feedback and push updates
- Once approved, maintainer will merge your PR

## Reporting Issues

### Bug Reports

Include:
- Python version & OS
- Reproduction steps
- Expected behavior
- Actual behavior
- Full error traceback
- Relevant code snippets

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative approaches considered
- Example code (if applicable)

## Documentation

### Contributing to Docs

```bash
# Build documentation locally
cd docs
sphinx-build -b html . _build
```

Documentation files are in Markdown or reStructuredText format.

## Large Contributions

For significant contributions:

1. **Discuss first:** Open an issue to discuss your proposal
2. **Design:** Get feedback on your design
3. **Implement:** Follow the workflow above
4. **Review:** Engage with reviewers constructively

## Getting Help

- **Questions:** Open a Discussion (if available)
- **Bugs:** Open an Issue
- **Documentation:** Check existing docs first
- **Community:** Join our Discord/Slack (if available)

## Recognition

Contributors will be:
- Added to CONTRIBUTORS.md
- Mentioned in release notes
- Recognized in project metadata

---

Thank you for making FedVAE-Trans better! 🙏
