#!/bin/bash

# Define an array of alpha values
alphas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Define an array of corresponding model zip paths
models=(
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/3utyz5hm/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/b8gtlpb2/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/1x396jy5/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/m0gy5wtx/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/xl56sfyg/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/hfx1mh91/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/k5ly0drz/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/yf2e0akv/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/mo8zzxt1/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/k94uejv0/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_no_interrupt/SAC/x7h5vlc1/model.zip"
)

# Loop through all alpha values and their corresponding model paths
for i in "${!alphas[@]}"; do
    alpha=${alphas[$i]}
    model=${models[$i]}
    echo "Running simulation for alpha=$alpha with model $model"
    
    # Repeat the command 100 times for the current alpha and model
    # for (( j=0; j<100; j++ )); do
    #     python sim2.py $alpha R1 PPO $model
    # done
    python sim2.py $alpha R1 SAC $model
done
