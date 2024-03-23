from src.TaxationEnv import TaxationEnv
from src.ActionWrapper import NormalizeActionWrapper
from config import configure_model
from src.model import ModelManager
from wandb.integration.sb3 import WandbCallback
from src.wandb_callback import WandbLoggingCallback
from src.evaluate import evaluate_and_save_plots
import wandb
from src.utils import load_model_for_evaluation, save_config, RewardType, set_log_scale_xaxis, load_model_for_resume
from stable_baselines3.common.env_util import make_vec_env
import time
from stable_baselines3.common.vec_env import VecNormalize
import os
import signal
import sys
import random
import string
import numpy as np
import traceback
from matplotlib import pyplot as plt
import json
# Declare these variables globally so they can be accessed in the save_and_close function
global model_save_path, env_save_path, evaluation_mode, run_id, model, env

def save_and_close(model, env, model_save_path, env_save_path):
    print("Saving training model and environment...")
    model.save(model_save_path)
    env.save(env_save_path)
    print(f"Model and environment saved to {model_save_path} and {env_save_path}")
    print('\n')

def main():
    # Initialize model and env to None in case they are referenced before being assigned.
    global model_save_path, env_save_path, evaluation_mode, run_id, model, env
    model_save_path, env_save_path = None, None
    run = None
    evaluation_mode = False
    model, env = None, None

    try:
        # Get model configuration from user input
        config = configure_model()

        # Setup Weights & Biases if required
        run_id = None
        if config["use_wandb"]:
            if config.get("run_id_resume_training"):
                # Existing run_id implies resuming an existing run
                run_id = config["run_id_resume_training"]
                run = wandb.init(project="Gov_sim_tax", entity="econmarl-sims", id=run_id, resume="must")
            else:
                # Construct the name string based on the condition
                name_suffix = config['reward_type'] if config['alpha'] != 1 else ''
                run_name = f"{config['algorithm']}_{config['total_steps']}_steps_{config['policy_type']}_alpha={config['alpha']:.3f}_envs={config['n_envs']}_Tmax={config['action_space_upper_bound'][0]}_{name_suffix}"

                run = wandb.init(
                    project="Gov_sim_tax",
                    entity="econmarl-sims",
                    config=config,
                    #If alpha is equal to 1, don't use reward_type in the name:
                    name=run_name,
                    #name=f"{config['algorithm']}_{config['total_steps']}_steps_{config['policy_type']}_alpha={config['alpha']:.3f}_envs={config['n_envs']}_{config['reward_type']}",
                    sync_tensorboard=True,# to upload metrics to tensorboard
                )
                run_id = run.id
                # Define special metrics for wandb
                # wandb.define_metric("Average Reward per Episode", step_metric="Episode")
                # wandb.define_metric("Episode Length", step_metric="Episode")
                # wandb.define_metric("Final Gini Index", step_metric="Episode")
                # wandb.define_metric("Final T_max", step_metric="Episode")
                # wandb.define_metric("Final T_esp", step_metric="Episode")
                # wandb.define_metric("Final T_scale", step_metric="Episode")
                # wandb.define_metric("Final Wealth Mean", step_metric="Episode")
                # wandb.define_metric("Final Wealth Median", step_metric="Episode")

        elif config.get("run_id_for_evaluation"):
            evaluation_mode = True
            run_id = config["run_id_for_evaluation"]
        else:
            # If wandb is not used, generate a unique run ID with a string of 8 random characters
            run_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))



        # Set up the environment: if 
        if "n_envs" in config and config["n_envs"] >= 1:
            if not (config.get("run_id_resume_training")):
                # Create parallel environments if n_envs is greater than 1
                env = make_vec_env(
                    lambda: TaxationEnv(
                        use_wandb=config["use_wandb"],
                        pop_size=config["pop_size"],
                        num_states=config["num_states"],
                        inital_taxes_params=config["inital_taxes_params"],
                        consumptions_params=config["consumptions_params"],
                        returns_params=config["returns_params"],
                        episode_steps=config["episode_steps"],
                        percentiles=config["percentiles"],
                        alpha=config["alpha"],
                        reward_type=RewardType[config["reward_type"]],
                        exp_salaries_params=config["exp_salaries_params"],
                        wealth_init_params=config["wealth_init_params"],
                        action_space_lower_bound=np.array(config["action_space_lower_bound"]),
                        action_space_upper_bound=np.array(config["action_space_upper_bound"]),
                    ),
                    n_envs=config["n_envs"]
                )
                # Wrap the environments with VecNormalize for observation normalization
                env = VecNormalize(env, norm_obs=False, norm_reward=False)
                #env = NormalizeActionWrapper(env)
            # else:
            #     #Load env
            #     model_save_path = f"results/saved_models/{config['algorithm']}/{run_id}"
            #     env_save_path = model_save_path + 'vecnorm.pkl'
            #     if os.path.exists(model_save_path) and os.path.exists(env_save_path):
            #          env = VecNormalize.load(env_save_path, env)
            #     else:
            #         print(f"Environment to load not found as {env_save_path}")

        # Initialize the model using ModelManager
        tensorboard_log_path = f"results/wandb_log/{config['algorithm']}/{run_id}" if config["use_wandb"] else None

        if "n_envs" in config and config.get("n_envs", 1) > 0:
            unwrapped_env = env.envs[0].unwrapped  # Access the unwrapped environment of the first sub-environment
            percentiles_for_callback = unwrapped_env.percentiles
            episode_counter = unwrapped_env.episode_counter

        if not (config.get("run_id_for_evaluation") or config.get("run_id_resume_training")):
            # Save the config file only when not in resume/evaluation mode
            config_save_path = f"results/saved_models/{config['algorithm']}/{run_id}/config.json"
            os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
            save_config(config, config_save_path)
            print(f"Configuration saved to {config_save_path}")

        # Check if a run ID for evaluation is provided
        if config.get("run_id_for_evaluation"):
            # Load a TaxEnv environment with alpha and reward_type from the model's configuration in the run directory (if available) as 'config.json'
            if os.path.exists(f"results/saved_models/{config['algorithm']}/{run_id}/config.json"):
                with open(f"results/saved_models/{config['algorithm']}/{run_id}/config.json", 'r') as file:
                    config = json.load(file)
                    env = TaxationEnv(use_wandb=False, reward_type=RewardType[config["reward_type"]], alpha=config["alpha"])
            else:
                raise FileNotFoundError(f"No config.json for run ID {run_id} in any algorithm folder.")
            # Load the model using the provided run ID
            model = load_model_for_evaluation(run_id, config['algorithm'])

        else:
            # Create a directory to save the trained model and environment
            model_save_dir = f"results/saved_models/{config['algorithm']}/{run_id}"
            os.makedirs(model_save_dir, exist_ok=True)  # Ensure the directory exists
            model_save_path = os.path.join(model_save_dir, 'model.zip')  # Use 'model.zip' as the filename
            env_save_path = os.path.join(model_save_dir, 'vecnorm.pkl')  # Use 'vecnorm.pkl' as the filename

            callbacks = [
                WandbCallback(gradient_save_freq=500, model_save_path=f"results/saved_models/{config['algorithm']}/{run_id}", verbose=2, model_save_freq=1000),
                WandbLoggingCallback(verbose=0, log_freq=500, percentiles=percentiles_for_callback)
                ] if config["use_wandb"] else []

            if config.get("run_id_resume_training"):
                model, env = load_model_for_resume(config["run_id_resume_training"])

                remaining_timesteps = config["total_steps"] - config["steps_done"]

                # Start timing the training process
                start_time = time.time()
                model.learn(total_steps=remaining_timesteps, callback=callbacks)

                # Calculate the total training time
                training_duration = time.time() - start_time
                print(f"Training duration: {training_duration:.4f} s")


            else:
                # Initialize the model using ModelManager
                model_manager = ModelManager(config["algorithm"], config["policy_type"], env, tensorboard_log=tensorboard_log_path, config=config)
                model = model_manager.create_model()
                # Train the model if no run ID is provided

                # Start timing the training process
                start_time = time.time()

                model.learn(total_timesteps=config["total_steps"], callback=callbacks) 

                # Calculate the total training time
                training_duration = time.time() - start_time
                print(f"Training duration: {training_duration:.4f} s")

            # Save the trained model and environment
            save_and_close(model, env, model_save_path, env_save_path)

        # Evaluate the trained model on a dummy env
        if not evaluation_mode:
            env_test = TaxationEnv(use_wandb=False, reward_type=RewardType[config["reward_type"]], alpha=config["alpha"])
        else:
            env_test = env
        evaluate_and_save_plots(model, env_test, n_eval_episodes=1, plot_path=f"results/saved_models/{config['algorithm']}/{run_id}/sim_plots/", algorithm=config["algorithm"])

        # if config.get("n_envs", 1) > 0:
        #     #unwrappe 1 env
        #     env.envs[0].unwrapped.show_statistics(mode='single', file_path=f"results/saved_models/{config['algorithm']}/{run_id}/plots/")
        # else:
        #     env.show_statistics(mode='single', file_path=f"results/saved_models/{config['algorithm']}/{run_id}/plots/")

        # Finish the wandb run if it was used
        if config["use_wandb"] and run is not None:
            run.finish()

        # Close the environment
        env.close()

    except KeyboardInterrupt:
        print('\n')
        if not evaluation_mode:
            print("Training interrupted by user.")
            if model is not None and env is not None:
                save_and_close(model, env, model_save_path, env_save_path)
        else:
            print("Simulation interrupted by user.")
    except Exception as e:
        print('\n')
        print(f'An unexpected error occurred: {e}')
        print('\n')
        traceback.print_exc()
        if model is not None and env is not None and not evaluation_mode:
            save_and_close(model, env, model_save_path, env_save_path)
    finally:
        if config.get("use_wandb", True) and run_id and not evaluation_mode:
            wandb.finish()
            print('\n')
            print(f"Wandb run {run_id} finished.")
            print('\n')
        if env is not None:
            env.close()
        print('Environment closed.')
        print('\n')


if __name__ == "__main__":
    main()