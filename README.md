# TaxationPolicyRL

## Overview
TaxationPolicyRL is a Reinforcement Learning (RL) project focused on simulating and optimizing taxation policies using various RL algorithms. The project utilizes a custom Gym environment to model the taxation system and applies algorithms like PPO, SAC, and TD3 to optimize policy decisions.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps
1. Clone the Repository
   ```bash
   git clone https://github.com/bancaditalia/EconMARL-sims.git
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

### Evaluation
- The project includes an evaluation script to assess the performance of the trained model.

## Project Structure
- `src/`: Source code for the project including the RL models and utilities.
- `TaxationEnv.py`: Custom Gym environment for the taxation policy simulation.
- `model.py`: Model manager for initializing various RL models.
- `config.py`: Configuration script for setting up model parameters and options.
- `requirements.txt`: List of dependencies for the project.
- `README.md`: This file, containing project information and instructions.
