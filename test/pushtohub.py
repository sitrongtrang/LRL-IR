import os
import json
from datasets import Dataset, DatasetInfo, concatenate_datasets
import pandas as pd

# Path to your JSON files
json_folder = "dq_vi_economic"
dataset_name = "IR4LRL/vietnamese-economic-query-document"
batch_size = 1000

data_in_batch = []
counter = 0

batches = []
# Convert to format that pandas can read
for filename in os.listdir(json_folder):
    file_path = os.path.join(json_folder, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)  # Load JSON
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {filename}: {e}")
            continue
        
        print(filename)
        data = {k:[v] for k,v in data.items()}
        
        data_in_batch.append(data)
        counter += 1
        
        if counter >= batch_size:
            counter = 0
            
            combined_data = {}
            for item in data_in_batch:
                for key, value in item.items():
                    if key not in combined_data:
                        combined_data[key] = value
                    else:
                        combined_data[key].extend(value)
                        
            batch = Dataset.from_dict(combined_data)
            
            batches.append(batch)
            
            data_in_batch = []
            
combined_data = {}
for item in data_in_batch:
    for key, value in item.items():
        if key not in combined_data:
            combined_data[key] = value
        else:
            combined_data[key].extend(value)
            
batch = Dataset.from_dict(combined_data)
batches.append(batch)

full_dataset = concatenate_datasets(batches)
full_dataset.push_to_hub(dataset_name)