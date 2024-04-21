# marl-demandresponse
# Frequency Regulation Simulator

Welcome to the Frequency Regulation Simulator, an advanced open-source platform designed to address the critical challenge of frequency regulation through demand response, executed at the granularity of seconds. This simulator leverages the robust OpenAI Gym framework, making it a potent tool for researchers and practitioners aiming to innovate within the domain of demand response management.

For more detailed insights into the theoretical framework and experimental validation, refer to our research paper: [Read the paper](https://arxiv.org/abs/2301.02593).

The original version of the code can be found at https://github.com/maivincent/marl-demandresponse-original

## Features
- **Flexible Simulation**: 
  - **Diverse Building Models**: Simulate energy consumption and thermal dynamics in various types of buildings. Each building can be modeled with specific thermal characteristics such as wall conductance, thermal mass, and insulation quality, reflecting real-world diversity.
  - **Customizable Components**: The simulator's architecture allows for the addition of custom elements like battery dynamics, enabling researchers to explore the impacts of different energy storage solutions on demand response.
  - **Advanced AC Modeling**: Air conditioners are simulated with detailed characteristics including cooling capacity, performance coefficients, and dynamic constraints such as compressor lockout to realistically model their operational limitations.
  - **Regulation Signal Adaptation**: The simulation includes a complex model of the regulation signal, incorporating high-frequency variations to mimic the intermittency of renewable energy sources. This feature challenges the system to maintain stability under realistic and dynamic conditions.
- **OpenAI Gym Integration**: Fully compatible with the OpenAI Gym framework, our simulator provides a familiar and accessible environment for conducting machine learning research.
- **Decentralized Agents**: The simulator includes two sophisticated decentralized agents trained via Multi-Agent Proximal Policy Optimization (MA-PPO):
  - **Agent with Hand-Engineered Communication**: This agent operates based on a meticulously crafted communication strategy designed to optimize coordination and efficiency.
  - **Agent with Dynamic Communication (TarMAC)**: Utilizing Targeted Multi-Agent Communication (TarMAC), this agent dynamically learns the most effective data-sharing strategies in response to environmental stimuli.
- **Baseline**: In addition to our advanced agents, the simulator also evaluates against several baseline control strategies:
  - **Bang-bang Controller (BBC)**: This decentralized algorithm operates by turning the AC on when the air temperature exceeds a target and off when it falls below, controlling the temperature near-optimally when lockout duration is zero. However, it struggles with signal tracking due to high-frequency variations.
  - **Greedy Myopic**: A centralized approach that operates like solving a knapsack problem at each timestep, choosing ACs based on their immediate value to temperature regulation versus power consumption. This method does not plan for the future, quickly depleting available AC units due to lockout constraints.
  - **Model Predictive Control (MPC)**: This centralized strategy models the environment and forecasts actions over a set horizon to maximize rewards. While theoretically optimal, MPC's performance degrades with increasing number of agents and extended time horizons due to its computational complexity.
- **Fully Controllable Interface**: Our simulator features a highly interactive and controllable interface, divided into several functionalities:
  - **Manual Action Overwrite**: Users can manually override the decisions made by the algorithm at any point during the simulation. This allows for experimental adjustments and real-time interaction.

    ![Manual Action Overwrite Example](https://i.ibb.co/PGBdqqL/image-Interface1.png)
  - **Individual House Monitoring**: The interface enables detailed observation and monitoring of individual houses, providing granular data on each unit's performance and status.

    ![Individual House Monitoring Example](https://i.ibb.co/HnswFZx/image-Interface2.png)
  - **Free Navigation Through Timesteps**: Users can move freely across any timestep in the simulation, facilitating thorough analysis and review of different scenarios and outcomes.



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
