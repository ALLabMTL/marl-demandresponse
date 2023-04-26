# marl-demandresponse

The original version of the code can be found at https://github.com/maivincent/marl-demandresponse-original

## Quickstart
1. Clone the repository
2. Navigate to the root of the project then inside the `server/` folder. 
3. Install the virtual environment and activate it
    ```bash
    python3.9 -m venv .venv
    source ./.venv/bin/activate # or ./.venv/Scripts/activate.bat on Windows
    ```
4. Install the requirements
    ```bash
    pip install -r requirements.txt # Python 3.9 is required
    ```
5. Run the code
    ```bash
    python app/main.py
    ```
6. In a new terminal, navigate back up to the root and then inside the `client/` folder. 
7. Install the dependencies with `npm i` 
8. Run the client with `npm start` 
9. The client should be running and you should be able to access the web page at `http://localhost:4200` 

### Troubleshooting
- Python 3.9 not installed
    - Install python 3.9 with `sudo apt install python3.9`, `brew install python3.9` or manually on Windows
- CVXPY solver errors 
    - You need an institutional license to use gurobi, cplex or mosek.
- `WARNING:  MARLconfig.json not found, using default config.`
    - Make sure you are running `main.py` from `server/app` and not `server/`

## Quickstart (VSCode, server only)
0. Ignore errors related to missing packages such as pylint, mypy or pydocstyle. 
1. Open command palette (F1) and choose Python: Create environnement > Venv. Choose Python version 3.9. This will take a while as it installs all the reqiurements (5 minutes on a two core laptop with no cached packages)
2. Open a new terminal, make sure the prompt says `.venv`

## Development
VScode is recommended for development. Default settings are provided in the `.vscode` folder, most notably the following settings are provided:
- `"python.formatting.provider": "black"` and `"editor.formatOnSave": true` to format the code on save with black
- `"python.linting.pylintEnabled": true` to lint the code with pylint

## Debug the code

```bash
python -m debugpy --listen localhost:5679 --wait-for-client app/main.py
# then attach the debugger in VScode (F5)
```

## Run linters
Inside the `server/app` folder, run the following commands:
```bash
isort -i . # Run first
autoflake -i -r . --remove-unused-variables --remove-rhs-for-unused-variables --ignore-init-module-imports --remove-all-unused-imports # (Optional, not installed by default)
flake8 ./ --statistics --ignore E501 # (Optional, not installed by default, install optional dependency flake8-bugbear)
pylint .
mypy . --ignore-missing-imports --install-types
black .
```

## Generating documentation

```bash
cd server/docs
sphinx-apidoc -f -o source/api/ ../app/
make html # .\make.bat html on Windows
python -m http.server -d build/html
python -m webbrowser -t "http://localhost:8000"
```
### Troubleshooting
- sphynx-apidoc not found
    - Make sure your are in your virtual environment
    - On Ubuntu, run `sudo apt install python3-sphinx`
- `make` not found
    - On Windows, run instead.

## Generating UML diagrams
Requirements:
- pyreverse (included in pylint, in requirements.txt)
- [plantuml](https://plantuml.com/download)
For one class at a time:
```bash
set PLANTUML_LIMIT_SIZE=18908
pyreverse app -o plantuml -m n
plantuml *.plantuml -progress
# resulting png file likely huge, consider one class at a time such as:
# pyreverse [...] --class=app.services.experiment_manager.ExperimentManager
```
