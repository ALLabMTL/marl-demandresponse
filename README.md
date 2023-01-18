# marl-demandresponse projet 4

## Quickstart

1. Clone the repository
2. Install the virtual environment and activate it
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install the requirements
    ```bash
    pip install -r requirements.txt
    ```
4. Run the code
    ```bash
    python .\main.py --exp 10 --agent_type ppo --no_wandb --render
    ```

## Development
VScode is recommended for development. Default settings are provided in the `.vscode` folder, most notably the following settings are provided:
- `"python.formatting.provider": "autopep8"` and `"editor.formatOnSave": true` to format the code on save with black
- `"python.linting.pylintEnabled": true` to lint the code with pylint
### Run the tests

```bash
python -m pytest
```

### Run linters

```bash
isort -i $(git ls-files '*.py')
pylint $(git ls-files '*.py')
mypy $(git ls-files '*.py') --ignore-missing-imports --install-types
autopep8 --in-place $(git ls-files '*.py')
```

