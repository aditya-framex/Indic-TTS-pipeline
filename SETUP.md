# Setup

This repo excludes large model files, environments, and generated
outputs (see .gitignore). To get the project running after cloning:


To get the project running after cloning:

1. Get the models folder:
   - Ask the project owner for the `models/` folder directly
     (transferred via drive/USB/direct copy — not downloaded from
     Hugging Face), and place it in the project root.

2. Set up the Python environment:
   
   python3 -m venv indicf5-env
   source indicf5-env/bin/activate
   
   pip install -r requirements.txt
   pip install ./IndicF5
