
"""

Pipeline stages
----------------
1. Extract audio          -> ffmpeg, 16kHz mono WAV
2. Voice Activity Detect  -> Silero-VAD, finds all speech-only regions
3. Speaker isolation      -> resemblyzer speaker embedding similarity
                              against a reference clip (if provided),
                              else largest diarized cluster (pyannote,
                              optional, only if no reference is given)
4. Quality scoring         -> SNR estimate + spectral-flatness (music/noise
                              rejection) + loudness sanity check
5. Window selection        -> merge adjacent good segments from the same
                              speaker into an 8-10s window, pick the
                              highest scoring one
6. Export                  -> ffmpeg slice -> MP3


Install
-------
    pip install torch torchaudio silero-vad-utils resemblyzer librosa \
                soundfile pydub numpy
    # ffmpeg must be installed and on PATH
    # optional, only needed for --diarize-only:
    pip install pyannote.audio
"""

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class Segment:
    start: float
    end: float
    score: float = 0.0
    speaker_sim: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


# --------------------------------------------------------------------------
# Stage 1: audio extraction
# --------------------------------------------------------------------------

def extract_audio(video_path: str, out_wav: str, sr: int = 16000) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", str(sr),
        "-vn", out_wav,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --------------------------------------------------------------------------
# Stage 2: VAD
# --------------------------------------------------------------------------

def run_vad(wav_path: str, sr: int = 16000) -> List[Segment]:
    """Silero-VAD via torch.hub. Returns speech segments in seconds."""
    import torch

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    (get_speech_timestamps, _, read_audio, *_rest) = utils

    wav = read_audio(wav_path, sampling_rate=sr)
    raw_segments = get_speech_timestamps(
        wav, model, sampling_rate=sr,
        min_speech_duration_ms=300,
        # keep this small: we WANT to preserve genuine short pauses as
        # candidate cut points rather than bridging over them here.
        min_silence_duration_ms=150,
    )
    return [Segment(s["start"] / sr, s["end"] / sr) for s in raw_segments]


def merge_micro_gaps(segments: List[Segment], max_gap: float = 0.12) -> List[Segment]:
    """
    Bridge only *very* short gaps (e.g. a stop-consonant closure inside a
    word) so we don't fragment a single word into multiple segments.
    Deliberately small so real pauses (which are valid cut points) survive.
    """
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s.start)
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg.start - merged[-1].end <= max_gap:
            merged[-1] = Segment(merged[-1].start, seg.end)
        else:
            merged.append(seg)
    return merged


# --------------------------------------------------------------------------
# Stage 3: speaker isolation (only relevant if a reference voice is given;
# safe to skip entirely for your current single-speaker videos)
# --------------------------------------------------------------------------

def load_speaker_encoder():
    from resemblyzer import VoiceEncoder
    return VoiceEncoder()


def get_reference_embedding(encoder, reference_wav: str) -> np.ndarray:
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(reference_wav)
    return encoder.embed_utterance(wav)


def score_speaker_similarity(
    encoder, full_wav_path: str, segments: List[Segment], ref_embedding: np.ndarray, sr: int = 16000
) -> None:
    from resemblyzer import preprocess_wav

    audio, file_sr = sf.read(full_wav_path)
    assert file_sr == sr

    for seg in segments:
        start_sample = int(seg.start * sr)
        end_sample = int(seg.end * sr)
        chunk = audio[start_sample:end_sample]
        if len(chunk) < sr * 0.5:
            seg.speaker_sim = 0.0
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, chunk, sr)
            wav = preprocess_wav(tmp.name)
        os.unlink(tmp.name)
        emb = encoder.embed_utterance(wav)
        seg.speaker_sim = float(np.dot(emb, ref_embedding) /
                                 (np.linalg.norm(emb) * np.linalg.norm(ref_embedding) + 1e-8))


# --------------------------------------------------------------------------
# Stage 4: quality scoring (SNR + speech-vs-music + loudness)
# --------------------------------------------------------------------------

def speech_modulation_score(chunk: np.ndarray, sr: int) -> float:
    """
    Classic speech/music discrimination cue: speech has a strong amplitude
    modulation peak around 3-6 Hz (the syllabic rate). Music and steady
    background sound don't share that peak, even when they're quiet and
    "clean" by SNR standards. Returns a 0-1 'speechiness' score — higher
    means more likely to be clear, unaccompanied speech.

    This is a lightweight signal-processing feature (no pretrained model,
    no download needed) so it drops in anywhere librosa/numpy already work.
    """
    if len(chunk) < sr:  # need ~1s of audio to get a meaningful spectrum
        return 0.5

    envelope = np.abs(chunk).astype(np.float64)
    win = max(1, int(sr * 0.01))  # 10ms smoothing
    envelope = np.convolve(envelope, np.ones(win) / win, mode="same")
    envelope = envelope - np.mean(envelope)

    n = len(envelope)
    spec = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    band_mask = (freqs > 0.5) & (freqs < 20)
    speech_mask = (freqs >= 3) & (freqs <= 6)

    total_energy = np.sum(spec[band_mask]) + 1e-10
    speech_band_energy = np.sum(spec[speech_mask])

    ratio = speech_band_energy / total_energy
    return float(np.clip(ratio / 0.35, 0, 1))


def _read_chunk(full_wav_path: str, seg: Segment, sr: int) -> np.ndarray:
    audio, file_sr = sf.read(full_wav_path)
    assert file_sr == sr
    start_sample = int(seg.start * sr)
    end_sample = int(seg.end * sr)
    return audio[start_sample:end_sample].astype(np.float32)


def quality_score(full_wav_path: str, seg: Segment, sr: int = 16000) -> float:
    chunk = _read_chunk(full_wav_path, seg, sr)
    if len(chunk) < sr * 0.5:
        return 0.0

    # --- crude SNR estimate: energy of loud frames vs quiet frames ---
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    frames = librosa.util.frame(chunk, frame_length=frame_len, hop_length=hop).T
    energies = np.mean(frames ** 2, axis=1) + 1e-10
    sorted_e = np.sort(energies)
    noise_floor = np.mean(sorted_e[: max(1, len(sorted_e) // 10)])
    signal_level = np.mean(sorted_e[-max(1, len(sorted_e) // 10):])
    snr_db = 10 * np.log10(signal_level / noise_floor)

    # --- speech-vs-music/background cue (see docstring above) ---
    speechiness = speech_modulation_score(chunk, sr)

    # --- loudness sanity: penalize near-silent or clipped chunks ---
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    peak = float(np.max(np.abs(chunk)) + 1e-8)
    clipping_penalty = 1.0 if peak < 0.98 else 0.5

    snr_component = np.clip(snr_db / 40.0, 0, 1)
    loudness_component = np.clip(rms * 20, 0, 1)

    return float(
        0.40 * snr_component
        + 0.40 * speechiness          # dominant signal for "no background music"
        + 0.20 * loudness_component
    ) * clipping_penalty


# --------------------------------------------------------------------------
# Stage 5: window selection — always cut on real pauses
# --------------------------------------------------------------------------

def build_candidate_windows(
    segments: List[Segment], min_dur: float, max_dur: float,
    sim_threshold: float, max_bridge_gap: float = 0.5,
) -> Tuple[List[Segment], List[Segment]]:
    """
    Build [min_dur, max_dur] windows by combining runs of consecutive VAD
    segments. A window's start/end are always a real detected speech
    boundary (a genuine pause), never an arbitrary timestamp.

    Gaps between combined segments up to `max_bridge_gap` are allowed
    *inside* a window (natural breathing pauses); larger gaps stop the run,
    since that likely signals a topic break or dead air not worth including.

    Returns (candidates, long_segments) — long_segments are speech runs
    with no internal pause, longer than max_dur, handled separately by
    refine_long_segment() since there's no VAD boundary to align to.
    """
    good = sorted([s for s in segments if s.speaker_sim >= sim_threshold], key=lambda s: s.start)

    candidates: List[Segment] = []
    long_segments: List[Segment] = []
    n = len(good)

    for i in range(n):
        j = i
        while j < n:
            if j > i and good[j].start - good[j - 1].end > max_bridge_gap:
                break
            window_start = good[i].start
            window_end = good[j].end
            duration = window_end - window_start
            if duration > max_dur:
                break
            if duration >= min_dur:
                candidates.append(Segment(window_start, window_end))
            j += 1

    for seg in good:
        if seg.duration > max_dur:
            long_segments.append(seg)

    return candidates, long_segments


def refine_long_segment(
    full_wav_path: str, seg: Segment, min_dur: float, max_dur: float,
    sr: int = 16000, from_end: bool = False,
) -> Optional[Segment]:
    """
    For a continuous speech run longer than max_dur with no real VAD pause
    inside it, find the quietest short moment (closest thing to a breath or
    micro-pause) within the valid [min_dur, max_dur] range and cut there —
    instead of truncating at a fixed offset, which risks landing mid-word.
    """
    audio, file_sr = sf.read(full_wav_path)
    assert file_sr == sr

    if not from_end:
        lo = int((seg.start + min_dur) * sr)
        hi = int(min(seg.end, seg.start + max_dur) * sr)
    else:
        lo = int(max(seg.start, seg.end - max_dur) * sr)
        hi = int((seg.end - min_dur) * sr)

    if hi <= lo:
        return None

    frame_len = int(0.05 * sr)  # 50ms
    hop = int(0.01 * sr)
    chunk = audio[lo:hi].astype(np.float32)
    if len(chunk) < frame_len:
        return None

    frames = librosa.util.frame(chunk, frame_length=frame_len, hop_length=hop).T
    energies = np.mean(frames ** 2, axis=1)
    quietest_frame = int(np.argmin(energies))
    cut_sample = lo + quietest_frame * hop + frame_len // 2
    cut_time = cut_sample / sr

    if not from_end:
        return Segment(seg.start, cut_time)
    else:
        return Segment(cut_time, seg.end)


# --------------------------------------------------------------------------
# Stage 6: export
# --------------------------------------------------------------------------

def export_audio(video_or_wav_path: str, seg: Segment, out_path: str) -> None:
    """Always exports MP3 (libmp3lame)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_or_wav_path,
        "-ss", f"{seg.start:.3f}",
        "-t", f"{seg.duration:.3f}",
        "-ac", "1", "-ar", "44100",
        "-c:a", "libmp3lame", "-q:a", "2",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --------------------------------------------------------------------------
# Interactive file picker (unchanged from your version)
# --------------------------------------------------------------------------

UPLOAD_DIR = "uploads"
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm")
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac")


def _scan_files(extensions: Tuple[str, ...]) -> List[str]:
    dirs_to_scan = [".", UPLOAD_DIR]
    found = []
    for d in dirs_to_scan:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(extensions):
                path = os.path.join(d, name) if d != "." else name
                found.append(path)
    seen = set()
    unique = []
    for f in found:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def pick_file(label: str, extensions: Tuple[str, ...], optional: bool = False) -> Optional[str]:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    while True:
        candidates = _scan_files(extensions)
        print(f"\n--- Select {label} ---")
        if candidates:
            for i, f in enumerate(candidates, 1):
                print(f"  [{i}] {f}")
        else:
            print("  (no matching files found in current directory or ./uploads)")

        print(f"  [u] Upload a new file (scp it into ./{UPLOAD_DIR}/ from your laptop, then choose this)")
        print(f"  [p] Type a full file path directly")
        if optional:
            print(f"  [s] Skip — only one speaker in the video, no need to isolate a target voice")

        choice = input("Choice: ").strip().lower()

        if optional and choice == "s":
            return None

        if choice == "u":
            print(f"\nFrom your LOCAL laptop terminal, run something like:")
            print(f"  scp -i /path/to/key.pem /local/path/to/file "
                  f"<user>@<ec2-ip>:~/{UPLOAD_DIR}/")
            input("Press Enter once the upload is finished to rescan...")
            continue

        if choice == "p":
            typed = input("Full path to file: ").strip()
            if os.path.isfile(typed):
                return typed
            print(f"File not found: {typed}")
            continue

        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]

        print("Invalid choice, try again.")


def interactive_config() -> "argparse.Namespace":
    print("=" * 60)
    print(" Voice Clip Extraction — Interactive Mode")
    print("=" * 60)

    input_path = pick_file("source video", VIDEO_EXTS)
    print("\nA reference sample helps the script identify WHICH voice to isolate,")
    print("when the video has more than one speaker (e.g. an interviewer + guest,")
    print("or multiple actors in a scene). If your video has only ONE speaker")
    print("throughout, you can skip this and it'll just pick the clearest segment.")
    reference_path = pick_file(
        "reference voice sample (clean audio of target speaker) — optional",
        AUDIO_EXTS, optional=True,
    )

    out_name = "audio.mp3"

    ns = argparse.Namespace(
        input=input_path,
        reference=reference_path,
        out=out_name,
        min_dur=8.0,
        max_dur=10.0,
        sim_threshold=0.75,
        keep_wav=False,
    )
    ref_desc = ns.reference if ns.reference else "(none — quality-only selection)"
    print(f"\nRunning with: input={ns.input}, reference={ref_desc}, out={ns.out}\n")
    return ns


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="Path to source video (mp4/mkv/etc.). Omit for interactive mode.")
    ap.add_argument("--reference", default=None,
                     help="Path to a clean reference audio (wav/mp3) of the target speaker. "
                          "Omit if the video has only one speaker.")
    ap.add_argument("--out", default="audio.mp3", help="Output path (always exported as MP3)")
    ap.add_argument("--min-dur", type=float, default=8.0)
    ap.add_argument("--max-dur", type=float, default=10.0)
    ap.add_argument("--sim-threshold", type=float, default=0.75)
    ap.add_argument("--keep-wav", action="store_true")
    args = ap.parse_args()

    if not args.input:
        args = interactive_config()
    elif not args.out.lower().endswith(".mp3"):
        args.out += ".mp3"

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "extracted.wav")
        print(f"[1/6] Extracting audio from {args.input} ...")
        extract_audio(args.input, wav_path)

        print("[2/6] Running VAD ...")
        raw_segments = run_vad(wav_path)
        speech_segments = merge_micro_gaps(raw_segments, max_gap=0.12)
        print(f"      Found {len(speech_segments)} speech segments (pauses preserved as cut points)")

        print("[3/6] Speaker filtering ...")
        if args.reference:
            encoder = load_speaker_encoder()
            ref_emb = get_reference_embedding(encoder, args.reference)
            score_speaker_similarity(encoder, wav_path, speech_segments, ref_emb)
            effective_sim_threshold = args.sim_threshold
        else:
            print("      No reference provided — single-speaker mode, skipping filtering")
            for seg in speech_segments:
                seg.speaker_sim = 1.0
            effective_sim_threshold = 0.0

        print("[4/6] Building pause-aligned candidate windows ...")
        candidates, long_segments = build_candidate_windows(
            speech_segments, args.min_dur, args.max_dur, effective_sim_threshold
        )
        for seg in long_segments:
            for from_end in (False, True):
                refined = refine_long_segment(wav_path, seg, args.min_dur, args.max_dur, from_end=from_end)
                if refined is not None:
                    candidates.append(refined)

        if not candidates:
            print("No candidate windows found. Try lowering --min-dur, or check that the "
                  "video has at least one continuous clean-speech stretch of that length.")
            sys.exit(1)

        print(f"      {len(candidates)} candidate window(s) generated")

        print("[5/6] Scoring candidate windows for clarity / background music ...")
        for c in candidates:
            c.score = quality_score(wav_path, c)
        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0]
        print(f"      Best window: {best.start:.2f}s - {best.end:.2f}s "
              f"(duration {best.duration:.2f}s, quality score {best.score:.3f})")

        print(f"[6/6] Exporting to {args.out} ...")
        export_audio(args.input, best, args.out)

        if args.keep_wav:
            keep_path = os.path.splitext(args.out)[0] + "_full.wav"
            import shutil
            shutil.copy(wav_path, keep_path)
            print(f"      Kept intermediate WAV at {keep_path}")

    print("Done.")
    
main()



# STAGE 2 -- IndicF5 TTS notebook code 


# (IPython magics are invalid syntax in a .py file). They are one-time environment setup; run once:
#   pip install transformers==4.49.0 accelerate==0.33.0
#   pip install /home/ubuntu/indicf5_package_source
#   pip install soundfile jiwer openai-whisper
#   pip install torchcodec
#   pip install sentencepiece
#   pip install IndicTransToolkit
#   pip install mosestokenizer indic-nlp-library nltk
import nltk
nltk.download("punkt")
from transformers import AutoModel

repo_id = "/home/ubuntu/models/indicf5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
model = model.to("cuda")
print(next(model.parameters()).device)
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

EN_INDIC_CKPT = "/home/ubuntu/models/indictrans2-en-indic"
INDIC_EN_CKPT = "/home/ubuntu/models/indictrans2-indic-en"
INDIC_INDIC_CKPT = "/home/ubuntu/models/indictrans2-indic-indic"

INDICCONFORMER_CKPT = "/home/ubuntu/models/indic-conformer"
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


def select_voice() -> str:
    """Automatically select the single reference voice file (no human input)."""
    voice_path = "audio.mp3"
    if not os.path.isfile(voice_path):
        raise FileNotFoundError(
            f"Expected reference voice file not found: '{os.path.abspath(voice_path)}'"
        )
    print(f"Using voice: {os.path.basename(voice_path)}")
    return voice_path


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


def translate_with_indictrans2(text: str, src_lang_code: str, tgt_lang_code: str, as_list: bool = False):
    """Translate text using IndicTrans2. If as_list=True, returns a list of translated sentences."""
    src_flores = ISO_TO_FLORES.get(src_lang_code)
    tgt_flores = ISO_TO_FLORES.get(tgt_lang_code)
    if src_flores is None:
        raise ValueError(f"No FLORES-200 mapping for source language '{src_lang_code}'.")
    if src_flores == tgt_flores:
        return [text] if as_list else text          # <-- changed

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

    return translations if as_list else " ".join(translations)   # <-- changed


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


def get_ref_text(voice_path: str, source_lang: str) -> str:
    """Transcribe the reference audio in its own language (A).

    The transcript is returned as-is and is NOT translated, because ref_text must
    match what is actually spoken in the reference audio.
    """
    print(f"\nTranscribing '{voice_path}' (source language: {source_lang})...")

    if source_lang == "en":
        transcript = _transcribe_with_whisper(voice_path)
    else:
        transcript = _transcribe_with_indicconformer(voice_path, source_lang)

    transcript = transcript.strip()
    print(f"Reference transcript ({source_lang}): {transcript}")
    return transcript


# --- run the automated setup ---
ref_audio_path = select_voice()
source_lang_code = select_source_language()   # language A: the clip's language
target_lang_code = select_target_language()   # language B: the language to speak
ref_text = get_ref_text(ref_audio_path, source_lang_code)   # transcript in A, no translation

voice_name = os.path.splitext(os.path.basename(ref_audio_path))[0]
print(f"\nReady: voice='{voice_name}', ref language='{source_lang_code}', output language='{target_lang_code}'")
print(f"ref_text ({source_lang_code}): {ref_text}")

# ---- NEW: translate the transcript A -> B, one sentence per list item ----
target_sentences = translate_with_indictrans2(
    ref_text, source_lang_code, target_lang_code, as_list=True
)
print(f"\nTranslated into {target_lang_code} ({len(target_sentences)} sentences):")
for i, s in enumerate(target_sentences, start=1):
    print(f"  {i}. {s}")


import os
import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000
GAP_SECONDS = 0.3  # silence between sentences

output_dir = f"dubbed_{source_lang_code}_to_{target_lang_code}_{voice_name}"
os.makedirs(output_dir, exist_ok=True)

clips = []
for i, sentence in enumerate(target_sentences, start=1):
    audio = model(
        sentence,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,          # transcript in A, matches the reference audio
    )
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    clips.append(np.array(audio, dtype=np.float32))
    print(f"[{i}/{len(target_sentences)}] Generated")

# Join all sentences into one file
gap = np.zeros(int(SAMPLE_RATE * GAP_SECONDS), dtype=np.float32)
parts = []
for idx, clip in enumerate(clips):
    parts.append(clip)
    if idx < len(clips) - 1:
        parts.append(gap)
dubbed = np.concatenate(parts)

full_path = f"{output_dir}/dubbed_audio.wav"
sf.write(full_path, dubbed, samplerate=SAMPLE_RATE)
print(f"\nSaved: {full_path}")


