import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from IndicTransToolkit.processor import IndicProcessor
from mosestokenizer import MosesSentenceSplitter
from nltk import sent_tokenize
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EN_INDIC_CKPT = "ai4bharat/indictrans2-en-indic-1B"
INDIC_EN_CKPT = "ai4bharat/indictrans2-indic-en-1B"
INDIC_INDIC_CKPT = "ai4bharat/indictrans2-indic-indic-1B" 

BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Only the 5 languages you need, mapped to FLORES codes -> 2-letter ISO
# (2-letter code needed by indic_nlp_library / mosestokenizer for sentence splitting)
FLORES_TO_ISO = {
    "eng_Latn": "en",
    "hin_Deva": "hi",
    "kan_Knda": "kn",
    "tam_Taml": "ta",
    "tel_Telu": "te",
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def initialize_model_and_tokenizer(ckpt_dir):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model = model.to(DEVICE)
    model.half()
    model.eval()
    return tokenizer, model

# ---------------------------------------------------------------------------
# Sentence splitting (needed before batch translation)
# ---------------------------------------------------------------------------
def split_sentences(input_text, lang):
    if lang == "eng_Latn":
        with MosesSentenceSplitter(FLORES_TO_ISO[lang]) as splitter:
            sents_moses = splitter([input_text])
        sents_nltk = sent_tokenize(input_text)
        input_sentences = sents_nltk if len(sents_nltk) < len(sents_moses) else sents_moses
        input_sentences = [s.replace("\xad", "") for s in input_sentences]
    else:
        input_sentences = sentence_split(
            input_text, lang=FLORES_TO_ISO[lang], delim_pat=DELIM_PAT_NO_DANDA
        )
    return input_sentences

# ---------------------------------------------------------------------------
# Core batch translation
# ---------------------------------------------------------------------------
def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(
            batch, truncation=True, padding="longest",
            return_tensors="pt", return_attention_mask=True,
        ).to(DEVICE)
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs, use_cache=True, min_length=0, max_length=256,
                num_beams=5, num_return_sequences=1, do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        generated_tokens = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True,
        )
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)
        del inputs
        torch.cuda.empty_cache()
    return translations

# ---------------------------------------------------------------------------
# Router: picks the right model pair automatically based on src/tgt
# ---------------------------------------------------------------------------
class TranslationPipeline:
    def __init__(self):
        print("Loading en<->indic model...")
        self.en_indic_tok, self.en_indic_model = initialize_model_and_tokenizer(EN_INDIC_CKPT)
        print("Loading indic<->en model...")
        self.indic_en_tok, self.indic_en_model = initialize_model_and_tokenizer(INDIC_EN_CKPT)
        print("Loading indic<->indic model...")
        self.indic_indic_tok, self.indic_indic_model = initialize_model_and_tokenizer(INDIC_INDIC_CKPT)
        self.ip = IndicProcessor(inference=True)
        print("All models loaded.")

    def translate(self, text, src_lang, tgt_lang):
        if src_lang == tgt_lang:
            return text
        sentences = split_sentences(text, src_lang)

        if src_lang == "eng_Latn":
            tok, model = self.en_indic_tok, self.en_indic_model
        elif tgt_lang == "eng_Latn":
            tok, model = self.indic_en_tok, self.indic_en_model
        else:
            tok, model = self.indic_indic_tok, self.indic_indic_model

        translated = batch_translate(sentences, src_lang, tgt_lang, model, tok, self.ip)
        return " ".join(translated)

    def translate_lines(self, lines, src_lang, tgt_lang):
        """Translate a list of already-separate lines/sentences in batches."""
        if src_lang == tgt_lang:
            return lines

        if src_lang == "eng_Latn":
            tok, model = self.en_indic_tok, self.en_indic_model
        elif tgt_lang == "eng_Latn":
            tok, model = self.indic_en_tok, self.indic_en_model
        else:
            tok, model = self.indic_indic_tok, self.indic_indic_model

        return batch_translate(lines, src_lang, tgt_lang, model, tok, self.ip)

# ---------------------------------------------------------------------------
# Test run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = TranslationPipeline()

    test_cases = [
        ("नमस्ते, आप कैसे हैं?", "hin_Deva", "eng_Latn"),
        ("Hello, how are you?", "eng_Latn", "hin_Deva"),
        ("नमस्ते, आप कैसे हैं?", "hin_Deva", "kan_Knda"),
        ("நீங்கள் எப்படி இருக்கிறீர்கள்?", "tam_Taml", "tel_Telu"),
    ]

    for text, src, tgt in test_cases:
        result = pipeline.translate(text, src, tgt)
        print(f"\n[{src} -> {tgt}]")
        print(f"IN:  {text}")
        print(f"OUT: {result}")
