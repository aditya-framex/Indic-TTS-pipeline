# Setup

This repo excludes large model files, environments, and generated
outputs (see .gitignore). To get the project running after cloning:


To get the project running after cloning:

1. Set up the Python environment:
   python3 -m venv indicf5-env
   source indicf5-env/bin/activate
   pip install -r requirements.txt
   pip install ./IndicF5

2. Download the required models:
   python3 allmod_down.py
