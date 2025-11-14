# ResearcherAI Book - Code Testing Infrastructure

This directory contains comprehensive testing for all code examples in the book to ensure they work correctly for readers.

## Overview

We use a **multi-layer testing approach**:

1. **Jupyter Notebooks** - Manual testing and verification (you can see outputs)
2. **Automated pytest** - CI/CD integration (runs on every commit)
3. **Docker** - Isolated environment testing
4. **GitHub Actions** - Prevents broken code from being deployed

## Directory Structure

```
tests/
├── notebooks/              # Jupyter notebooks for manual testing
│   ├── test_01_vector_databases.ipynb
│   ├── test_02_hands_on_tutorial.ipynb
│   └── ...
├── automated/              # Automated pytest tests
│   └── test_code_examples.py
├── requirements.txt        # All dependencies
├── Dockerfile              # Docker environment
└── README.md              # This file
```

## Quick Start

### Option 1: Test with Jupyter Notebooks (Recommended for Manual Testing)

**Install dependencies:**
```bash
cd tests
pip install -r requirements.txt
```

**Launch Jupyter:**
```bash
jupyter lab
```

**Run notebooks:**
- Open `notebooks/test_01_vector_databases.ipynb`
- Run all cells (Kernel → Restart & Run All)
- Verify all outputs match expectations

**What to check:**
- ✓ No errors or exceptions
- ✓ Outputs match book examples
- ✓ Assertions pass

### Option 2: Run Automated Tests (CI/CD)

**Run all tests:**
```bash
cd tests
pytest automated/ -v
```

**Run specific test:**
```bash
pytest automated/test_code_examples.py::TestVectorDatabases::test_faiss_vector_store -v
```

**Expected output:**
```
===== test session starts =====
tests/automated/test_code_examples.py::TestVectorDatabases::test_embedding_creation PASSED
tests/automated/test_code_examples.py::TestVectorDatabases::test_cosine_similarity PASSED
tests/automated/test_code_examples.py::TestVectorDatabases::test_faiss_vector_store PASSED
tests/automated/test_code_examples.py::TestHandsOnTutorial::test_transform_function PASSED
...
===== 10 passed in 45.23s =====
```

### Option 3: Test with Docker (Isolated Environment)

**Build and run:**
```bash
cd tests
docker build -t researcherai-tests .
docker run researcherai-tests
```

This ensures tests run in a clean environment exactly like CI/CD.

## GitHub Actions Integration

Tests run automatically on every push:

**Workflow:** `.github/workflows/test-code.yml`

**What it does:**
1. Runs automated pytest suite
2. Executes all Jupyter notebooks
3. Validates Python syntax in markdown
4. Reports results in pull request

**View results:**
- Go to GitHub → Actions tab
- Click on latest workflow run
- See test results and coverage

**Safety:** Code won't deploy if tests fail!

## Testing Workflow

### Before Committing New Code Examples

1. **Write code in book** (`docs/data-foundations.md`)
2. **Test manually** with Jupyter notebooks
3. **Add automated test** to `automated/test_code_examples.py`
4. **Run tests locally:**
   ```bash
   cd tests
   pytest automated/ -v
   ```
5. **Commit** if all tests pass
6. **CI/CD runs** automatically on push
7. **Deploy** only if CI/CD passes

## What Gets Tested

### Vector Databases (Part 1)
- ✓ Embedding creation with SentenceTransformers
- ✓ Cosine similarity calculations
- ✓ FAISS vector store implementation
- ✓ Qdrant vector store (optional, if running)

### Hands-On Tutorial (Part 2)
- ✓ CSV data loading
- ✓ SPARQL CONSTRUCT transform function
- ✓ Knowledge graph construction (28 triples)
- ✓ SPARQL SELECT queries
- ✓ RDF to NetworkX conversion
- ✓ Graph serialization to Turtle format

### Knowledge Graphs (Part 3)
- ✓ RDF triple creation
- ✓ Turtle format parsing
- ✓ SPARQL queries
- ✓ Graph visualization

### Code Quality
- ✓ Python syntax validation
- ✓ Import statement correctness
- ✓ Function signatures match
- ✓ No runtime errors

## Troubleshooting

### Notebook won't execute
```
Error: Kernel died before replying
```

**Solution:**
- Install kernel: `python -m ipykernel install --user`
- Restart Jupyter

### Pytest import errors
```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Solution:**
```bash
pip install -r requirements.txt
```

### Docker build fails
```
Error response from daemon: No space left on device
```

**Solution:**
```bash
docker system prune -a
docker build -t researcherai-tests .
```

### Tests timeout
Some tests download models (sentence-transformers). First run may take 2-5 minutes.

**Solution:** Be patient or increase timeout in pytest.ini

## Adding New Tests

### For New Code Examples in Book

1. **Add to notebook:**
```python
# In notebooks/test_XX_new_feature.ipynb
def test_new_feature():
    # Your code from book
    result = new_function()
    assert result == expected
    print("✓ Test PASSED")
```

2. **Add to automated tests:**
```python
# In automated/test_code_examples.py
class TestNewFeature:
    def test_new_function(self):
        """Test: New function from Chapter X"""
        result = new_function()
        assert result == expected
```

3. **Run tests:**
```bash
pytest automated/test_code_examples.py::TestNewFeature -v
```

## CI/CD Configuration

**GitHub Actions:** `.github/workflows/test-code.yml`

**Triggers:**
- Push to master/main
- Pull requests
- Manual dispatch

**Jobs:**
1. `test-code-examples` - Runs pytest suite
2. `test-notebooks` - Executes all notebooks
3. `code-quality` - Validates syntax
4. `summary` - Aggregates results

**Failure handling:**
- Tests must pass before deployment
- Failed tests block deployment automatically
- Detailed error reports in GitHub Actions logs

## Best Practices

1. **Always test manually first** with Jupyter notebooks
2. **Verify outputs** match book examples exactly
3. **Add automated tests** for every code block
4. **Run full suite** before committing
5. **Check CI/CD** before deploying
6. **Keep dependencies updated** (requirements.txt)

## Dependencies

All dependencies are listed in `requirements.txt`:

```
numpy>=1.24.0
pandas>=2.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
qdrant-client>=1.7.0
rdflib>=7.0.0
neo4j>=5.14.0
networkx>=3.0
matplotlib>=3.7.0
pytest>=7.4.0
jupyter>=1.0.0
```

**Update dependencies:**
```bash
pip install --upgrade -r requirements.txt
```

## Support

If tests fail:

1. Check GitHub Actions logs
2. Run locally to reproduce
3. Review error messages
4. Fix code in book
5. Re-run tests
6. Commit fix

## Summary

This testing infrastructure ensures:
- ✓ All code examples work correctly
- ✓ Readers can run examples without errors
- ✓ Breaking changes are caught before deployment
- ✓ Book maintains high quality standards

**Remember:** Tests are your safety net. Use them!
