#!/bin/bash

# Define an array of alpha values
alphas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Define an array of corresponding model zip paths
models=(
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2_with_cv/PPO/8yamtvw0/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2_with_cv/PPO/89zhp49s/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2_with_cv/PPO/mlga69hn/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2_with_cv/PPO/740d15a9/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2_with_cv/PPO/pwjyoddw/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2_with_cv/PPO/kb6wxko3/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2_with_cv/PPO/cx1sl0s1/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2_with_cv/PPO/koyz36c8/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2_with_cv/PPO/jgo8xyzl/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2_with_cv/PPO/80xtoijv/model.zip"
    )

# Loop through all alpha values and their corresponding model paths
for i in "${!alphas[@]}"; do
    alpha=${alphas[$i]}
    model=${models[$i]}
    echo "Running simulation for alpha=$alpha with model $model"
    
    # Repeat the command 100 times for the current alpha and model
    for (( j=0; j<10; j++ )); do
        python sim.py $alpha R3 PPO $model
    done
done
