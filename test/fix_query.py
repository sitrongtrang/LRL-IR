import os
import json
import re

folder_path = 'dq_vi_economic'

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if "query" in data:
            title = data["title"]
            data["query"] = re.sub(rf'\b{re.escape(title)}\b', "", data["query"], flags=re.IGNORECASE).strip()

            title = title.replace('–', '-')
            data["query"] = re.sub(rf'\b{re.escape(title)}\b', "", data["query"], flags=re.IGNORECASE).strip()

            title = re.sub(r'\(.*?\)', '', title).strip()
            data["query"] = re.sub(rf'\b{re.escape(title)}\b', "", data["query"], flags=re.IGNORECASE).strip()
        
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

print("All JSON files have been updated.")
