import os, json
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client.handle_db
collection = db.documents

metadata_dir = "../metadata_json"

for file in os.listdir(metadata_dir):
    with open(os.path.join(metadata_dir, file)) as f:
        data = json.load(f)

    doc = {
        "_id": data["file_name"],
        "path": data["path"],
        "zone": "raw",
        "metadata": [
            {
                "type": "Descriptive",
                "properties": [
                    {"key": k, "value": str(v)}
                    for k, v in data.items()
                    if k not in ["file_name", "path"]
                ]
            }
        ]
    }

    collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
