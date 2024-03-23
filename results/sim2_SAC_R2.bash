#!/bin/bash

# Define an array of alpha values
alphas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Define an array of corresponding model zip paths
models=(
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/3grij8lc/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/tk7mehb6/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/3rhybxyh/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/w9bxzad0/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/9pzh3ifl/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/yk67xdqq/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/393mjn3o/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/htndec4u/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/zb8eonyq/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/9jpnexqo/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/x7h5vlc1/model.zip"
)

# Loop through all alpha values and their corresponding model paths
for i in "${!alphas[@]}"; do
    alpha=${alphas[$i]}
    model=${models[$i]}
    echo "Running simulation for alpha=$alpha with model $model"
    
    # Repeat the command 100 times for the current alpha and model
    # for (( j=0; j<100; j++ )); do
    #     python sim2.py $alpha R2 PPO $model
    # done
    python sim2.py $alpha R2 SAC $model
done
