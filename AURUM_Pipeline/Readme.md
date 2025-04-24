*** Create a Virtual Environment ****

source venv/bin/activate

***Create the profiles ***

python src/profile_columns.py --input_dir extracted_csv/ --output_dir profiles/

***Create the Graph using profiles ***

python src/build_ekg.py --profile_dir profiles/ --output ekg_graph/ekg.graphml


*** Visualize the Graph ***

python src/visualize_ekg.py --graph ekg_graph/ekg.graphml