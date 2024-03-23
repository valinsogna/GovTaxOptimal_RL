#!/bin/bash

# Define an array of alpha values
alphas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Define an array of corresponding model zip paths
models=(
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_R2_cv_no_interrup/SAC/akml1jei/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_R2_cv_no_interrup/SAC/sx4js4nv/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_R2_cv_no_interrup/SAC/0rqsvp0n/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_R2_cv_no_interrup/SAC/dm6wbzgs/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_R2_cv_no_interrup/SAC/0vohmbjf/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_R2_cv_no_interrup/SAC/25ref4lh/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_R2_cv_no_interrup/SAC/u40leibr/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_R2_cv_no_interrup/SAC/oetusztv/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_R2_cv_no_interrup/SAC/ym9f2fae/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_R2_cv_no_interrup/SAC/zrlcycv2/model.zip"
    )

# Loop through all alpha values and their corresponding model paths
for i in "${!alphas[@]}"; do
    alpha=${alphas[$i]}
    model=${models[$i]}
    echo "Running simulation for alpha=$alpha with model $model"
    
    # Repeat the command 100 times for the current alpha and model
    # for (( j=0; j<100; j++ )); do
    #     python sim.py $alpha R2 PPO $model
    # done
    python sim2.py $alpha R3 SAC $model
done
