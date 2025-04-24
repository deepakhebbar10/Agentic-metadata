import os, json
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client.handle_db
collection = db.documents

metadata_dir = "../metadata_json"

for file in os.listdir(metadata_dir):
    if not file.endswith(".json"):
        continue

    file_path = os.path.join(metadata_dir, file)
    print(f"📄 Processing: {file}")

    try:
        with open(file_path) as f:
            data = json.load(f)

        if "file_name" not in data or "path" not in data:
            print("⚠️ Missing required fields in:", file)
            continue

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

        result = collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        if result.upserted_id:
            print("✅ Inserted:", doc["_id"])
        else:
            print("♻️ Updated:", doc["_id"])

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
