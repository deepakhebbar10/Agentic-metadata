import os
from tika import parser
import json

input_dir = "data"
output_dir = "./metadata_json"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".pdf"):
        file_path = os.path.join(input_dir, file)
        parsed = parser.from_file(file_path)
        metadata = parsed.get('metadata', {})
        metadata['file_name'] = file
        metadata['path'] = file_path
        output_path = os.path.join(output_dir, file + ".json")
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
