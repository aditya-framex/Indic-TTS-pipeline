# Setup

This repo excludes large model files, environments, and generated
outputs (see .gitignore). To get the project running after cloning:

1. Download the required models:
   python3 allmod_down.py

2. Set up the Python environment:
   python3 -m venv indicf5-env
   source indicf5-env/bin/activate
   pip install -r requirements.txt
   git clone https://github.com/ai4bharat/IndicF5.git indicf5_package_source
   pip install ./indicf5_package_source

3. IndicF5/ and IndicTrans2/ are included with local modifications
   already applied — no separate clone needed for those.
