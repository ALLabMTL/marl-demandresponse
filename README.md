# marl-demandresponse projet 4

The previous code is in server/v0

## Quickstart

1. Clone the repository
2. Navigate to the root of the project then inside the `server/` folder. 
3. Install the virtual environment and activate it
    ```bash
    python3.9 -m venv .venv
    source ./.venv/bin/activate
    ```
4. Install the requirements
    ```bash
    pip install -r requirements.txt
    ```
5. Run the code
    ```bash
    python app/main.py
    ```
6. In a new terminal, navigate back up to the root and then inside the `client/` folder. 
7. Install the dependencies with `npm i` 
8. Run the client with `npm start` 
9. The client should be running and you should be able to access the web page at `http://localhost:4200` 

## Quickstart (VSCode, server only)
0. Ignore errors related to missing packages such as pylint, mypy or pydocstyle. 
1. Open command palette (F1) and choose Python: Create environnement > Venv. Choose Python version 3.9. This will take a while as it installs all the reqiurements (5 minutes on a two core laptop with no cached packages)
2. Open a new terminal, make sure the prompt says `.venv`

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
python -m debugpy --listen localhost:5600 --wait-for-client app/main.py
# then attach the debugger in VScode (F5)
```

### Run linters
Inside the `server/` folder, run the following commands:
```bash
isort -i $(git ls-files '*.py')
pylint $(git ls-files '*.py')
mypy $(git ls-files '*.py') --ignore-missing-imports --install-types
black $(git ls-files '*.py')
```

### Documentation

```bash
sphinx-apidoc -f -o docs/source/api/ .
cd docs
make html
python -m http.server -d build/html
python -m webbrowser -t "http://localhost:8000"
```
