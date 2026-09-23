"""
Run this once: python3 fix_indicf5_model.py
It edits /home/ubuntu/models/indicf5/model.py to stop reaching out to
Hugging Face for the vocoder and the vocab file, pointing both at your
already-downloaded local copies instead.
"""

MODEL_PY_PATH = "/home/ubuntu/models/indicf5/model.py"

with open(MODEL_PY_PATH, "r") as f:
    content = f.read()

# --- Fix 1: vocoder loading -------------------------------------------
old_vocoder_line = (
    'self.vocoder = torch.compile(load_vocoder(vocoder_name="vocos", '
    'is_local=False, device=device))'
)
new_vocoder_line = (
    'self.vocoder = torch.compile(load_vocoder(vocoder_name="vocos", '
    'is_local=True, local_path="/home/ubuntu/models/vocos-mel-24khz", device=device))'
)

if old_vocoder_line in content:
    content = content.replace(old_vocoder_line, new_vocoder_line)
    print("Fixed: vocoder loading now points to local folder.")
else:
    print("WARNING: couldn't find the exact vocoder line to replace — "
          "the file may already be edited, or the text doesn't match exactly. "
          "No change made for this part.")

# --- Fix 2: vocab.txt loading ------------------------------------------
old_vocab_line = (
    'vocab_path = hf_hub_download(config.name_or_path, filename="checkpoints/vocab.txt")'
)
new_vocab_line = (
    'vocab_path = "/home/ubuntu/models/indicf5/checkpoints/vocab.txt"'
)

if old_vocab_line in content:
    content = content.replace(old_vocab_line, new_vocab_line)
    print("Fixed: vocab.txt loading now points to local file.")
else:
    print("WARNING: couldn't find the exact vocab line to replace — "
          "the file may already be edited, or the text doesn't match exactly. "
          "No change made for this part.")

with open(MODEL_PY_PATH, "w") as f:
    f.write(content)

print("\nDone. Re-run your notebook cell now.")