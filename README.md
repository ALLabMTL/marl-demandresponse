# marl-demandresponse projet 4

## Quickstart

1. Clone the repository
2. Install the virtual environment and activate it
    ```bash
    python3.9 -m venv .venv
    source ./.venv/bin/activate
    ```
3. Install the requirements
    ```bash
    pip install -r requirements.txt
    ```
4. Run the code
    ```bash
    python main.py --exp 10 --agent_type ppo --no_wandb --render
    ```

## Development
VScode is recommended for development. Default settings are provided in the `.vscode` folder, most notably the following settings are provided:
- `"python.formatting.provider": "black"` and `"editor.formatOnSave": true` to format the code on save with black
- `"python.linting.pylintEnabled": true` to lint the code with pylint
### Run the tests

```bash
python -m pytest
```

### Debug the code

```bash
python -m debugpy --listen localhost:5678 --wait-for-client main.py --exp 10 --agent_type ppo --no_wandb
# then attach the debugger in VScode (F5)
```

### Run linters

```bash
isort -i $(git ls-files '*.py')
pylint $(git ls-files '*.py')
mypy $(git ls-files '*.py') --ignore-missing-imports --install-types
black $(git ls-files '*.py')
```

