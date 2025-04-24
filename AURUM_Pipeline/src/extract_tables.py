import os
import camelot
import argparse

def extract_tables(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".pdf"):
            tables = camelot.read_pdf(os.path.join(input_dir, file), pages='all')
            for i, table in enumerate(tables):
                out_path = os.path.join(output_dir, f"{file[:-4]}_table_{i}.csv")
                table.to_csv(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    extract_tables(args.input_dir, args.output_dir)
