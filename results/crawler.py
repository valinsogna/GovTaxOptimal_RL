import os
import json
import pandas as pd
import sys

def extract_model_paths(folder_path):
    # Lists to store the extracted data
    #r1_data = []
    r2_data = []

    # Walk through the directory
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            # Check for config.json file
            if file == 'config.json':
                config_path = os.path.join(subdir, file)
                
                # Read the config.json file
                with open(config_path, 'r') as config_file:
                    config = json.load(config_file)
                    
                    # Extract alpha and reward_type
                    alpha = config.get('alpha', None)
                    reward_type = config.get('reward_type', None)
                    model_path = os.path.join(subdir, 'model.zip')
                    
                    # Depending on the reward_type, add the data to the corresponding list
                    if reward_type == 'R1':
                       r1_data.append({'alpha': alpha, 'model_path': model_path})
                    elif reward_type == 'R2':
                        r2_data.append({'alpha': alpha, 'model_path': model_path})

    # Convert to DataFrames
    #r1_df = pd.DataFrame(r1_data).sort_values(by='alpha')
    r2_df = pd.DataFrame(r2_data).sort_values(by='alpha')

    # Save to CSV
    #r1_df.to_csv(parent_folder +'/r1_models.csv', index=False)
    r2_df.to_csv(parent_folder +'/r2_models.csv', index=False)

    print('CSV files created for R1 and R2 model paths.')

if __name__ == "__main__":
    #Parse command line arg and take alpha or defualt 0.5, and reward type or default R1 and the path where to pick the model
    if len(sys.argv) == 2:
        parent_folder = sys.argv[1]

    else:
        raise RuntimeError("You must provide parent_folder as a command line argument")
  
    # Call the function with the path to the SAC directory
    extract_model_paths(parent_folder)
