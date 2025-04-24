import os
import pandas as pd
from datasketch import MinHash
import argparse
import json

def profile_column(values):
    m = MinHash(num_perm=128)
    for val in values.dropna().astype(str):
        m.update(val.encode('utf8'))
    return list(m.hashvalues)

def profile_file(filepath):
    df = pd.read_csv(filepath)
    profiles = {}
    for col in df.columns:
        mh = profile_column(df[col])
        unique_ratio = df[col].nunique() / len(df)
        profiles[col] = {
            "minhash": [int(x) for x in mh],        # 👈 Convert MinHash hashvalues
            "unique_ratio": float(unique_ratio)     # 👈 Ensure it's a Python float
        }
    return profiles


def run(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            profiles = profile_file(os.path.join(input_dir, file))
            with open(os.path.join(output_dir, f"{file}.json"), "w") as f:
                json.dump(profiles, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)
