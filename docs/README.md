# Project Template
>*codeblocks are reversed due to a pycharm bug*

Use this template via `git clone https://github.com/OliverStoll/python-template`

## Prepare Project
Install the requirements and pre-commit hooks 
```bash
poetry install --no-root
pip install poetry
```

### Hide unnecessary folder from PyCharm
*Settings -> Editor -> File Types -> Ignore files and folders*
- poetry.lock
- .mypy_cache
- .pytest_cache
- .ruff_cache

## Use Project

### Run tests
```bash
pytest
```

### Serve documentation
```bash
sphinx-serve -h localhost
make html
sphinx-apidoc -t _templates -o source/ ../src
make clean
cd docs
```