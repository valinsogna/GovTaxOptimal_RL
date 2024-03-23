#!/bin/bash

# Define an array of alpha values
alphas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Define an array of corresponding model zip paths
models=(
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2without_cv/qljvym7d/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2without_cv/giw71dj6/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2without_cv/pmtapif3/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2without_cv/0d6sw6wn/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2without_cv/qety7zsf/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2without_cv/mha0uk7v/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2without_cv/2uyha1jo/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2without_cv/0jyom4a3/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2without_cv/fcve7l74/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2without_cv/lm79v4q7/model.zip"
    "/home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL/results/saved_models/Tmax=0.10_PPOnew_R2_with_cv/PPO/k69aent5/model.zip"
)

# Loop through all alpha values and their corresponding model paths
for i in "${!alphas[@]}"; do
    alpha=${alphas[$i]}
    model=${models[$i]}
    echo "Running simulation for alpha=$alpha with model $model"
    
    # Repeat the command 100 times for the current alpha and model
    for (( j=0; j<10; j++ )); do
        python sim.py $alpha R2 PPO $model
    done
done
