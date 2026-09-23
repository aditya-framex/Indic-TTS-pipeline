import csv
from translate_pipeline import TranslationPipeline

# --- Adjust these paths to match your actual eval set files ---
SENTENCE_FILES = {
    "eng_Latn": "eval_sets/english_eval.txt",
    "hin_Deva": "eval_sets/hindi_eval.txt",
    "kan_Knda": "eval_sets/kannada_eval.txt",
}

TIER1_PAIRS = [
    ("eng_Latn", "hin_Deva"),
    ("hin_Deva", "eng_Latn"),
    ("eng_Latn", "kan_Knda"),
    ("kan_Knda", "eng_Latn"),
    ("hin_Deva", "kan_Knda"),
    ("kan_Knda", "hin_Deva"),
]

LANG_NAMES = {"eng_Latn": "English", "hin_Deva": "Hindi", "kan_Knda": "Kannada"}


def load_sentences(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    sentences = {lang: load_sentences(path) for lang, path in SENTENCE_FILES.items()}
    for lang, sents in sentences.items():
        print(f"{LANG_NAMES[lang]}: {len(sents)} sentences loaded")

    pipeline = TranslationPipeline()

    for src_lang, tgt_lang in TIER1_PAIRS:
        src_name = LANG_NAMES[src_lang]
        tgt_name = LANG_NAMES[tgt_lang]
        out_fname = f"eval_output_{src_name}_to_{tgt_name}.csv"

        print(f"\nRunning {src_name} -> {tgt_name} ({len(sentences[src_lang])} sentences)...")

        with open(out_fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", f"source_{src_name}", f"translation_{tgt_name}", "accuracy_ok", "fluency_ok", "notes"])

            for i, sentence in enumerate(sentences[src_lang], start=1):
                translation = pipeline.translate(sentence, src_lang, tgt_lang)
                writer.writerow([i, sentence, translation, "", "", ""])

        print(f"Saved: {out_fname}")


if __name__ == "__main__":
    main()
