from src.utils import parse_number_list
import argparse
import json


# Function to load configuration from a JSON file
def load_config_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_user_input(prompt, default_value, type_cast=str):
    """ Helper function to get user input or use default value """
    user_input = input(prompt)
    return type_cast(user_input) if user_input else default_value

def configure_prompt():
    """ Function to configure the model based on user input or defaults """
    print("Configure RL Model. Press Enter to use default values.")

    # Add a prompt for the run ID
    run_id_for_evaluation = get_user_input("Enter the run ID for model evaluation (leave blank if not applicable): ", "")
    #run_id_resume_training = get_user_input("Enter the run ID for resuming training (leave blank if not applicable): ", "")

    if run_id_for_evaluation:
        # Only request minimal necessary info when a run_id is provided for evaluation
        algorithm = get_user_input("Enter Algorithm (e.g., PPO default, SAC, TD3): ", "PPO")
        #policy_type = get_user_input("Enter Policy Type (e.g., MlpPolicy default, CnnPolicy, MultiInputPolicy): ", "MlpPolicy")
      
        config = {
            "use_wandb": False,
            "algorithm": algorithm,
            "run_id_for_evaluation": run_id_for_evaluation,
        }
    else:
        run_id_resume_training = get_user_input("Enter the run ID for resuming training (leave blank if not applicable): ", "")
    # If run_id_resume_training is different from empty string, then enter this condition:
        if run_id_resume_training:
            algorithm = get_user_input("Enter Algorithm (e.g., PPO default, SAC, TD3): ", "PPO")
            policy_type = get_user_input("Enter Policy Type (e.g., MlpPolicy default, CnnPolicy, MultiInputPolicy): ", "MlpPolicy")
            total_steps = int(get_user_input("Enter Total Timesteps (e.g., 500_000  default): ", "500000"))
            steps_done = int(get_user_input("Enter Already Trained Timesteps (e.g., 0  default): ", "0"))
            use_wandb = get_user_input("Use Weights & Biases for logging? (yes default/no): ", "yes", str).lower() == "yes"
    
            config = {
                "use_wandb": use_wandb,
                "algorithm": algorithm,
                "policy_type": policy_type,
                "total_steps": total_steps,
                "steps_done": steps_done,
                "run_id_resume_training": run_id_resume_training,
            }
    
        else:
            # Full configuration needed for training a new model
            use_wandb = get_user_input("Use Weights & Biases for logging? (yes default/no): ", "yes", str).lower() == "yes"
            algorithm = get_user_input("Enter Algorithm (e.g., PPO default, SAC, TD3): ", "PPO")
            policy_type = get_user_input("Enter Policy Type (e.g., MlpPolicy default, CnnPolicy, MultiInputPolicy): ", "MlpPolicy")
            total_steps = int(get_user_input("Enter Total Timesteps (e.g., 500_000  default): ", "500000"))
            episode_steps = get_user_input("Enter Maximum Number of Steps per Episode (e.g., 1_000 default): ", "1000")
            n_envs = get_user_input("Enter the number of parallel environments (e.g., 1 for single environment): ", "1")
            alpha = get_user_input("Enter Alpha parameter for Convex combination of Gini index and average consumption (e.g., 1 default): ", "1")
            reward_type = get_user_input("Enter Reward Type (e.g., R1 default, R2): ", "R1")
            
            # Environment parameters
            pop_size = get_user_input("Enter Population Size (e.g., 1_000 default): ", "1000")
            num_states = get_user_input("Enter Number of States (e.g., 9 default): ", "9")
            percentiles_input = get_user_input("Enter Wealth Percentiles (e.g., 1, 10, 25, 50, 75, 90, 95, 99, 99.9 default): ", "1, 10, 25, 50, 75, 90, 95, 99, 99.9")
            inital_taxes_params_input= get_user_input("Enter Initial Tax Parameters (e.g. T_max=0.04,  T_esp=0.5, T_scale=0.5 default): ", "0.04, 0.5, 0.5")
            action_space_lower_bound = get_user_input("Enter Lower Bound for Action Space (e.g., T_max=0, T_esp=0, T_scale=0 default): ", "0, 0, 0")
            action_space_upper_bound = get_user_input("Enter Upper Bound for Action Space (e.g., T_max=0.99, T_esp=1, T_scale=1 default): ", "0.99, 1, 1")
            consumptions_params= get_user_input("Enter Default Consumption Parameters (e.g. η=0.8, K=5 default): ", "0.8, 5")
            returns_params_input= get_user_input("Enter Default Return Parameters (e.g. R_mean=0.04, R_std=0.2 default): ", "0.04, 0.2")
            exp_salaries_params = get_user_input("Enter Expected Salaries Parameters (e.g. mean=10, std=1 default): ", "10, 1")
            wealth_init_params = get_user_input("Enter Wealth Initialization Parameters (e.g. mean=12, std=2 default): ", "12, 2")
    
            # Model parameters
            learning_rate = get_user_input("Enter Learning Rate (e.g., 3e-4 default): ", "3e-4")
            schedule = get_user_input("Enter Learning Rate Schedule (e.g., const default, linear, exponential, cosine decays): ", "const")
        
            config = {
                "use_wandb": use_wandb if run_id_for_evaluation == "" else False,
                "algorithm": algorithm,
                "policy_type": policy_type,
                "total_steps": total_steps,
                "learning_rate": float(learning_rate),
                "pop_size": int(pop_size),
                "num_states": int(num_states),
                "inital_taxes_params": parse_number_list(inital_taxes_params_input),
                "consumptions_params": parse_number_list(consumptions_params),
                "returns_params": parse_number_list(returns_params_input),
                "percentiles": parse_number_list(percentiles_input),
                "episode_steps": int(episode_steps),
                "run_id_for_evaluation": run_id_for_evaluation,
                "schedule": schedule,
                "n_envs": int(n_envs),
                "alpha": float(alpha),
                "reward_type": reward_type,
                "exp_salaries_params": parse_number_list(exp_salaries_params),
                "wealth_init_params": parse_number_list(wealth_init_params),
                "action_space_lower_bound": parse_number_list(action_space_lower_bound),
                "action_space_upper_bound": parse_number_list(action_space_upper_bound),
            }

    return config



def configure_model():
    """ Function to configure the model based on user input or defaults """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run model with optional configuration file.')
    parser.add_argument('-f', '--config_file', help='Path to JSON configuration file.', type=str, default=None)
    args = parser.parse_args()

    # Check if a configuration file was provided
    if args.config_file:
        try:
            config = load_config_from_file(args.config_file)
            print("Configuration loaded from file:", config)
        except Exception as e:
            print(f"Failed to load configuration from {args.config_file}: {e}")
            config = configure_prompt()
    else:
        config = configure_prompt()
    print("Configuration:", config)

    return config
