#!/bin/bash

# Attiva tmux session chiamata valeria
#tmux a -t valeria

# # Avvia bash shell
# bash

# # Attiva l'ambiente conda specificato
# conda activate taxrl

# # Cambia directory
# cd /home/black-it-experiments/disk/tax-marl-experiments/EconMARL-sims/TaxationPolicyRL

# Assicurati che le cartelle config/ e output/ esistano
mkdir -p config
mkdir -p output

# Loop per eseguire lo script Python con diverse configurazioni e core
for i in {0..14}
do
  # Calcola il range di core da usare (0-3, 4-7, ...)
  core_start=$(( i * 4 ))
  core_end=$(( i * 4 + 3 ))

  # Costruisce la stringa del comando taskset
  taskset_cmd="taskset -c $core_start-$core_end"

  # Costruisce il comando completo e lo esegue, reindirizzando l'output
  cmd="$taskset_cmd python main.py -f config/config_$((i+15)).json > output/output_$((i+15)) 2>&1 &"

  echo "Esecuzione del comando: $cmd"
  eval $cmd
done

# tmux detach -s valeria