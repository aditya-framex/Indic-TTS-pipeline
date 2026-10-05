# Setup

This repo excludes large model files, environments, and generated
outputs (see .gitignore). To get the project running after cloning:

1. Get the models folder:
   - Download the `models/` folder from this Google Drive link:
     https://drive.google.com/file/d/1A_JwBmPS2v4hJamZw17yBaWXzhlNPCMm/view?usp=sharing
   - Extract it and place it in the project root.

2. Set up the Python environment:

   python3 -m venv indicf5-env
   source indicf5-env/bin/activate

   pip install -r requirements.txt
   pip install ./IndicF5
