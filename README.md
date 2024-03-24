# TaxationPolicyRL

## Overview
TaxationPolicyRL is a Reinforcement Learning (RL) project focused on simulating and optimizing wealth taxation policies using various RL algorithms. The project utilizes a custom Gym environment to model the taxation system and applies algorithms like PPO and SAC to optimize policy decisions.

This project was supervised by collaboration with researchers form Bank of Italy Aldo Glielmo and Valerio Astuti.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps
1. Clone the Repository
   ```bash
   git clone https://github.com/valinsogna/GovTaxOptimal_RL.git
   ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Configuration
- Modify the `config.py` to change model configurations, including the choice of RL algorithm, policy type, and total timesteps.
- Enable or disable Weights & Biases logging through the configuration prompts.

### Training the Model
- Run the main script to start training:
    ```bash
    python main.py
    ```
- Run the main script with settings configured in config.json file:
    ```bash
    python main.py -f config.json
    ```
### Evaluation
- The project includes an evaluation script to assess the performance of the trained model.

## Project Structure
- `src/`: Source code for the project including the RL models and utilities.
- `TaxationEnv.py`: Custom Gym environment for the taxation policy simulation.
- `model.py`: Model manager for initializing various RL models.
- `config.py`: Configuration script for setting up model parameters and options.
- `requirements.txt`: List of dependencies for the project.
- `README.md`: This file, containing project information and instructions.
- `results/`: Folder for results storage.
- `baseline/`: Comparable baseline algorithm Multi-Objective PSO.
