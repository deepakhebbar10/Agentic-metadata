import os, json
from tika import parser

INPUT_DIR = "../data"
OUTPUT_DIR = "../metadata_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for file_name in os.listdir(INPUT_DIR):
    if file_name.endswith(".pdf"):
        file_path = os.path.join(INPUT_DIR, file_name)
        parsed = parser.from_file(file_path)
        metadata = parsed.get("metadata", {})
        metadata["file_name"] = file_name
        metadata["path"] = file_path
        with open(os.path.join(OUTPUT_DIR, file_name + ".json"), "w") as f:
            json.dump(metadata, f, indent=2)