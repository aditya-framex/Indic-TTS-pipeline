"""
Run this once: python3 fix_indicconformer_model.py
Edits /home/ubuntu/models/indic-conformer/model_onnx.py so its
from_pretrained() method uses a local folder directly instead of
always calling snapshot_download (which forces a Hugging Face lookup).
"""

MODEL_PY_PATH = "/home/ubuntu/models/indic-conformer/model_onnx.py"

with open(MODEL_PY_PATH, "r") as f:
    content = f.read()

old_line = "        loc = snapshot_download(repo_id=pretrained_model_name_or_path, token=token)"
new_block = (
    "        import os\n"
    "        if os.path.isdir(pretrained_model_name_or_path):\n"
    "            loc = pretrained_model_name_or_path\n"
    "        else:\n"
    "            loc = snapshot_download(repo_id=pretrained_model_name_or_path, token=token)"
)

if old_line in content:
    content = content.replace(old_line, new_block)
    with open(MODEL_PY_PATH, "w") as f:
        f.write(content)
    print("Fixed: from_pretrained now uses the local folder directly when given one.")
else:
    print("WARNING: couldn't find the exact line to replace — "
          "the file may already be edited, or spacing doesn't match exactly. "
          "No change made.")
