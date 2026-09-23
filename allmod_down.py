"""
Run this ONCE, on your own machine, while logged into Hugging Face
(use a fresh, non-leaked token this time — set it as an environment
variable, don't paste it in code: os.environ["HF_TOKEN"] = "..." or
better, set it in your terminal before running Python).

Downloads every model your pipeline uses into a single "models" folder
that you will move into your GitHub-tracked project (but NOT commit
the actual files to git — see step 3 below).
"""
from huggingface_hub import snapshot_download

MODELS_TO_DOWNLOAD = {
    "indicf5": "ai4bharat/IndicF5",
    "indictrans2-en-indic": "ai4bharat/indictrans2-en-indic-1B",
    "indictrans2-indic-en": "ai4bharat/indictrans2-indic-en-1B",
    "indictrans2-indic-indic": "ai4bharat/indictrans2-indic-indic-1B",
    "indic-conformer": "ai4bharat/indic-conformer-600m-multilingual",
}

for local_name, repo_id in MODELS_TO_DOWNLOAD.items():
    print(f"Downloading {repo_id} ...")
    snapshot_download(repo_id=repo_id, local_dir=f"./models/{local_name}")
    print(f"  -> saved to ./models/{local_name}\n")

print("All models downloaded. The 'models' folder is now fully self-contained.")