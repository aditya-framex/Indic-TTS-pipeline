import os
import glob
import torch
import torchaudio
import whisper  # kept only as a fallback for English reference clips
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from mosestokenizer import MosesSentenceSplitter
from nltk import sent_tokenize
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA

VOICES_DIR = "voices"  # folder containing your 10 reference voice clips

# Restricted to the 4 languages we have matching evaluation sentence lists for.
LANGUAGE_CHOICES = {
    "1": ("hi", "Hindi"),
    "2": ("kn", "Kannada"),
    "3": ("te", "Telugu"),
    "4": ("ta", "Tamil"),
}

# Source-language menu adds English, since reference clips might be English
# and IndicConformer cannot transcribe English at all (it only covers the
# 22 official Indian languages).
SOURCE_LANGUAGE_CHOICES = {
    **LANGUAGE_CHOICES,
    "5": ("en", "English"),
}

EN_INDIC_CKPT = "ai4bharat/indictrans2-en-indic-1B"
INDIC_EN_CKPT = "ai4bharat/indictrans2-indic-en-1B"
INDIC_INDIC_CKPT = "ai4bharat/indictrans2-indic-indic-1B"  # distilled, lighter on memory

INDICCONFORMER_CKPT = "ai4bharat/indic-conformer-600m-multilingual"
INDICCONFORMER_TARGET_SR = 16000
INDICCONFORMER_DECODING = "ctc"  # or "rnnt" for slightly higher accuracy, slower

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FLORES_TO_ISO = {
    "eng_Latn": "en", "hin_Deva": "hi", "kan_Knda": "kn",
    "tam_Taml": "ta", "tel_Telu": "te",
}
ISO_TO_FLORES = {v: k for k, v in FLORES_TO_ISO.items()}

_model_cache = {}  # checkpoint name -> (tokenizer, model); only loads what a given run needs
_indicconformer_model = None  # lazy-loaded singleton, only if an Indic source clip is used
_whisper_model = None  # lazy-loaded singleton, only if an English source clip is used


def select_voice(voices_dir: str = VOICES_DIR) -> str:
    """List available reference voices and let the user pick one."""
    voice_files = sorted(
        glob.glob(os.path.join(voices_dir, "*.mp3"))
        + glob.glob(os.path.join(voices_dir, "*.wav"))
    )
    if not voice_files:
        raise FileNotFoundError(f"No .mp3/.wav files found in '{voices_dir}'")

    print("Available voices for cloning:")
    for idx, path in enumerate(voice_files, start=1):
        print(f"  {idx}. {os.path.basename(path)}")

    while True:
        choice = input(f"Select a voice (1-{len(voice_files)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(voice_files):
            selected = voice_files[int(choice) - 1]
            print(f"Selected: {os.path.basename(selected)}")
            return selected
        print("Invalid choice, try again.")


def select_target_language() -> str:
    """Ask which language the reference text should be translated into."""
    print("\nTarget language for reference text:")
    for key, (code_, name) in LANGUAGE_CHOICES.items():
        print(f"  {key}. {name} ({code_})")

    while True:
        choice = input(f"Select target language (1-{len(LANGUAGE_CHOICES)}): ").strip()
        if choice in LANGUAGE_CHOICES:
            code_, name = LANGUAGE_CHOICES[choice]
            print(f"Selected: {name} ({code_})")
            return code_
        print("Invalid choice, try again.")


def select_source_language() -> str:
    """Ask what language the selected reference voice clip is spoken in.

    IndicConformer needs the language passed in explicitly (no auto-detection),
    and can't handle English at all, so this replaces Whisper's automatic
    language detection with an explicit choice.
    """
    print("\nWhat language is the reference voice clip spoken in?")
    for key, (code_, name) in SOURCE_LANGUAGE_CHOICES.items():
        print(f"  {key}. {name} ({code_})")

    while True:
        choice = input(f"Select source language (1-{len(SOURCE_LANGUAGE_CHOICES)}): ").strip()
        if choice in SOURCE_LANGUAGE_CHOICES:
            code_, name = SOURCE_LANGUAGE_CHOICES[choice]
            print(f"Selected: {name} ({code_})")
            return code_
        print("Invalid choice, try again.")


def _select_checkpoint(src_flores: str, tgt_flores: str) -> str:
    if src_flores == "eng_Latn" and tgt_flores != "eng_Latn":
        return EN_INDIC_CKPT
    if tgt_flores == "eng_Latn" and src_flores != "eng_Latn":
        return INDIC_EN_CKPT
    return INDIC_INDIC_CKPT


def _load_model(checkpoint: str):
    if checkpoint not in _model_cache:
        print(f"Loading IndicTrans2 model '{checkpoint}' (first time only)...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint, trust_remote_code=True,
            attn_implementation="eager", low_cpu_mem_usage=True,
        )
        model = model.to(DEVICE)
        model.half()
        model.eval()
        _model_cache[checkpoint] = (tokenizer, model)
    return _model_cache[checkpoint]


def _split_sentences(text: str, flores_lang: str):
    """IndicTrans2 translates best sentence-by-sentence rather than on a whole
    multi-clause utterance at once."""
    if flores_lang == "eng_Latn":
        with MosesSentenceSplitter(FLORES_TO_ISO[flores_lang]) as splitter:
            sents_moses = splitter([text])
        sents_nltk = sent_tokenize(text)
        sentences = sents_nltk if len(sents_nltk) < len(sents_moses) else sents_moses
        sentences = [s.replace("\xad", "") for s in sentences]
    else:
        sentences = sentence_split(text, lang=FLORES_TO_ISO[flores_lang], delim_pat=DELIM_PAT_NO_DANDA)
    return sentences or [text]


def has_excessive_repetition(text: str, max_consecutive_repeats: int = 3) -> bool:
    """Flag translations where the same word repeats back-to-back -- a sign of decoder degeneration."""
    words = text.split()
    run_length = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run_length += 1
            if run_length > max_consecutive_repeats:
                return True
        else:
            run_length = 1
    return False


def _generate(sentences, src_flores, tgt_flores, tokenizer, model, ip,
              num_beams=5, repetition_penalty=1.2, no_repeat_ngram_size=3):
    batch = ip.preprocess_batch(sentences, src_lang=src_flores, tgt_lang=tgt_flores)
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs, use_cache=True, min_length=0, max_length=256,
            num_beams=num_beams, num_return_sequences=1, do_sample=False,
            repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
        )
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ip.postprocess_batch(decoded, lang=tgt_flores)


def translate_with_indictrans2(text: str, src_lang_code: str, tgt_lang_code: str) -> str:
    """Translate text using IndicTrans2. src/tgt are whisper-style codes, e.g. 'ta', 'hi'."""
    src_flores = ISO_TO_FLORES.get(src_lang_code)
    tgt_flores = ISO_TO_FLORES.get(tgt_lang_code)
    if src_flores is None:
        raise ValueError(f"No FLORES-200 mapping for detected source language '{src_lang_code}'.")
    if src_flores == tgt_flores:
        return text

    checkpoint = _select_checkpoint(src_flores, tgt_flores)
    tokenizer, model = _load_model(checkpoint)
    ip = IndicProcessor(inference=True)
    sentences = _split_sentences(text, src_flores)

    translations = _generate(sentences, src_flores, tgt_flores, tokenizer, model, ip)

    for idx, (sentence, translation) in enumerate(zip(sentences, translations)):
        if has_excessive_repetition(translation):
            print(f"  Warning: degenerate translation for sentence {idx + 1}, retrying with stricter decoding...")
            retried = _generate(
                [sentence], src_flores, tgt_flores, tokenizer, model, ip,
                num_beams=1, repetition_penalty=1.5, no_repeat_ngram_size=2,
            )
            if has_excessive_repetition(retried[0]):
                print(f"  Still degenerate after retry: {retried[0]!r}")
            translations[idx] = retried[0]

    return " ".join(translations)


def _load_indicconformer():
    global _indicconformer_model
    if _indicconformer_model is None:
        print(f"Loading {INDICCONFORMER_CKPT} (first time only)...")
        _indicconformer_model = AutoModel.from_pretrained(
            INDICCONFORMER_CKPT,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
    return _indicconformer_model


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper large-v3 (first time only, English fallback)...")
        _whisper_model = whisper.load_model("large-v3")
    return _whisper_model


def _transcribe_with_indicconformer(voice_path: str, source_lang: str) -> str:
    model = _load_indicconformer()
    wav, sr = torchaudio.load(voice_path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != INDICCONFORMER_TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=INDICCONFORMER_TARGET_SR)
        wav = resampler(wav)
    wav = wav.to(DEVICE)
    with torch.no_grad():
        transcription = model(wav, source_lang, INDICCONFORMER_DECODING)
    return transcription


def _transcribe_with_whisper(voice_path: str) -> str:
    model = _load_whisper()
    result = model.transcribe(voice_path, language="en")
    return result["text"]


def get_ref_text(voice_path: str, target_lang: str, source_lang: str) -> str:
    """Transcribe the reference audio, then translate into the target language with IndicTrans2.

    source_lang is now supplied explicitly (via select_source_language) instead of
    being auto-detected, since IndicConformer has no built-in language detection.
    """
    print(f"\nTranscribing '{voice_path}' (source language: {source_lang})...")

    if source_lang == "en":
        transcript = _transcribe_with_whisper(voice_path)
    else:
        transcript = _transcribe_with_indicconformer(voice_path, source_lang)

    print(f"Transcript: {transcript}")

    translated = translate_with_indictrans2(transcript, source_lang, target_lang)
    print(f"Translated reference text ({target_lang}): {translated}")
    return translated


# --- run the automated setup ---
ref_audio_path = select_voice()
source_lang_code = select_source_language()
target_lang_code = select_target_language()
ref_text = get_ref_text(ref_audio_path, target_lang_code, source_lang_code)

voice_name = os.path.splitext(os.path.basename(ref_audio_path))[0]
print(f"\nReady: voice='{voice_name}', target_lang='{target_lang_code}'")
print(f"ref_text: {ref_text}")