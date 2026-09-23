import sys
import os
from translate_pipeline import TranslationPipeline
3
LANGUAGES = {
    "1": ("eng_Latn", "English"),
    "2": ("hin_Deva", "Hindi"),
    "3": ("kan_Knda", "Kannada"),
    "4": ("tam_Taml", "Tamil"),
    "5": ("tel_Telu", "Telugu"),
}

def choose_language(prompt_label):
    print(f"\n{prompt_label}")
    for key, (_, name) in LANGUAGES.items():
        print(f"  {key}. {name}")
    choice = input("Enter number: ").strip()
    while choice not in LANGUAGES:
        choice = input("Invalid choice, try again: ").strip()
    return LANGUAGES[choice]

def handle_typed_input(pipeline, src_code, src_name, tgt_code, tgt_name):
    print(f"\n[{src_name} -> {tgt_name}] Type your text (or 'back' to change languages, 'exit' to quit):")
    while True:
        text = input("> ").strip()
        if text.lower() == "exit":
            sys.exit(0)
        if text.lower() == "back":
            return
        if not text:
            continue
        result = pipeline.translate(text, src_code, tgt_code)
        print(f"Translation: {result}\n")

def handle_file_input(pipeline, src_code, src_name, tgt_code, tgt_name):
    file_path = input("\nEnter the full path to your text file: ").strip()
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("File is empty.")
        return

    print(f"Translating {len(lines)} lines from {src_name} to {tgt_name}...")
    translations = pipeline.translate_lines(lines, src_code, tgt_code)

    base, _ = os.path.splitext(file_path)
    out_path = f"{base}_{src_name}_to_{tgt_name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for src_line, tgt_line in zip(lines, translations):
            f.write(f"IN:  {src_line}\nOUT: {tgt_line}\n\n")

    print(f"Done. Output saved to: {out_path}")
    print("Open this file in the VS Code editor (not the terminal) to view non-English scripts correctly.\n")

def main():
    pipeline = TranslationPipeline()

    while True:
        src_code, src_name = choose_language("Select SOURCE language:")
        tgt_code, tgt_name = choose_language("Select TARGET language:")

        if src_code == tgt_code:
            print("Source and target are the same language — pick different ones.")
            continue

        print("\nHow do you want to provide input?")
        print("  1. Type a sentence")
        print("  2. Translate from a text file")
        mode = input("Enter number: ").strip()
        while mode not in ("1", "2"):
            mode = input("Invalid choice, enter 1 or 2: ").strip()

        if mode == "1":
            handle_typed_input(pipeline, src_code, src_name, tgt_code, tgt_name)
        else:
            handle_file_input(pipeline, src_code, src_name, tgt_code, tgt_name)

if __name__ == "__main__":
    main()