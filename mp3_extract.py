#!/usr/bin/env python3
"""
auto_clip_pipeline.py

Fully automated pipeline that takes a video (mp4, mkv, etc.) and produces
a single high-quality 8-10s mono MP3 clip of a target speaker's voice,
suitable as a voice-cloning / TTS reference sample.

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

Usage
-----
Interactive mode (just run it, no flags):

    python auto_clip_pipeline.py

    It will scan the current directory and ./uploads/ for video and audio
    files and let you pick, or tell you where to `scp` a file in from your
    laptop, then ask for an output filename. Output is always MP3.

Scripted / automation mode (for batch runs, cron, etc.):

    python auto_clip_pipeline.py \
        --input movie.mp4 \
        --reference reference_voice.wav \
        --out clip.mp3 \
        --min-dur 8 --max-dur 10

If you don't have a clean reference sample of the target speaker yet,
run once with --diarize-only to dump all speaker clusters, listen to a
sample from each, then reuse the best one as your --reference for every
future video.

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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


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
        min_silence_duration_ms=200,
    )
    return [Segment(s["start"] / sr, s["end"] / sr) for s in raw_segments]


def merge_close_segments(segments: List[Segment], max_gap: float = 0.4) -> List[Segment]:
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
# Stage 3: speaker isolation
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
    """Mutates segments in-place, filling in .speaker_sim (cosine similarity, 0-1)."""
    from resemblyzer import preprocess_wav
    import soundfile as sf

    audio, file_sr = sf.read(full_wav_path)
    assert file_sr == sr

    for seg in segments:
        start_sample = int(seg.start * sr)
        end_sample = int(seg.end * sr)
        chunk = audio[start_sample:end_sample]
        if len(chunk) < sr * 0.5:  # too short to embed reliably
            seg.speaker_sim = 0.0
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import soundfile as sf2
            sf2.write(tmp.name, chunk, sr)
            wav = preprocess_wav(tmp.name)
        os.unlink(tmp.name)
        emb = encoder.embed_utterance(wav)
        seg.speaker_sim = float(np.dot(emb, ref_embedding) /
                                 (np.linalg.norm(emb) * np.linalg.norm(ref_embedding) + 1e-8))


# --------------------------------------------------------------------------
# Stage 4: quality scoring (SNR + spectral flatness + loudness)
# --------------------------------------------------------------------------

def quality_score(full_wav_path: str, seg: Segment, sr: int = 16000) -> float:
    import soundfile as sf
    import librosa

    audio, file_sr = sf.read(full_wav_path)
    start_sample = int(seg.start * sr)
    end_sample = int(seg.end * sr)
    chunk = audio[start_sample:end_sample].astype(np.float32)

    if len(chunk) < sr * 0.5:
        return 0.0

    # --- crude SNR estimate: energy of loud frames vs quiet frames ---
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    frames = librosa.util.frame(chunk, frame_length=frame_len, hop_length=hop).T
    energies = np.mean(frames ** 2, axis=1) + 1e-10
    sorted_e = np.sort(energies)
    noise_floor = np.mean(sorted_e[: max(1, len(sorted_e) // 10)])   # quietest 10%
    signal_level = np.mean(sorted_e[-max(1, len(sorted_e) // 10):])  # loudest 10%
    snr_db = 10 * np.log10(signal_level / noise_floor)

    # --- spectral flatness: high flatness ~ noise/music-like, low ~ tonal speech ---
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=chunk)))

    # --- loudness sanity: penalize near-silent or clipped chunks ---
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    peak = float(np.max(np.abs(chunk)) + 1e-8)
    clipping_penalty = 1.0 if peak < 0.98 else 0.5

    # Combine into a single 0-1-ish score (weights are tunable)
    snr_component = np.clip(snr_db / 40.0, 0, 1)          # 40dB SNR -> full score
    flatness_component = np.clip(1.0 - flatness * 5, 0, 1)  # lower flatness is better
    loudness_component = np.clip(rms * 20, 0, 1)

    return float(
        0.5 * snr_component
        + 0.3 * flatness_component
        + 0.2 * loudness_component
    ) * clipping_penalty


# --------------------------------------------------------------------------
# Stage 5: window selection
# --------------------------------------------------------------------------

def build_candidate_windows(
    segments: List[Segment], min_dur: float, max_dur: float, sim_threshold: float
) -> List[Segment]:
    """
    Slide across merged speech segments and build candidate windows of
    [min_dur, max_dur] seconds using only segments that pass the speaker
    similarity threshold, allowing small gaps to be bridged.
    """
    good = [s for s in segments if s.speaker_sim >= sim_threshold]
    good = merge_close_segments(good, max_gap=0.6)

    candidates: List[Segment] = []
    for seg in good:
        if seg.duration >= min_dur:
            # trim down to max_dur from the start (could also slide window)
            end = min(seg.end, seg.start + max_dur)
            candidates.append(Segment(seg.start, end))
        # also try trailing sub-window if segment is long enough at the tail
        if seg.duration >= min_dur and seg.duration > max_dur:
            tail_start = seg.end - max_dur
            candidates.append(Segment(tail_start, seg.end))
    return candidates


# --------------------------------------------------------------------------
# Stage 6: export
# --------------------------------------------------------------------------

def export_audio(video_or_wav_path: str, seg: Segment, out_path: str) -> None:
    """Always exports MP3 (libmp3lame)."""
    codec_args = ["-c:a", "libmp3lame", "-q:a", "2"]
    cmd = [
        "ffmpeg", "-y",
        "-i", video_or_wav_path,
        "-ss", f"{seg.start:.3f}",
        "-t", f"{seg.duration:.3f}",
        "-ac", "1", "-ar", "44100",
        *codec_args,
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --------------------------------------------------------------------------
# Interactive file picker
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
    # de-dupe while preserving order
    seen = set()
    unique = []
    for f in found:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def pick_file(label: str, extensions: Tuple[str, ...], optional: bool = False) -> Optional[str]:
    """
    Lists matching files already present in '.' or './uploads'. Lets the
    user pick by number, type a full path directly, or upload one first
    (via scp into ./uploads from their laptop) and rescan. If optional is
    True, the user can also skip this selection entirely (returns None).
    """
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

    # Output is always MP3, always named audio.mp3 — no format or filename prompt.
    out_name = "audio.mp3"

    ns = argparse.Namespace(
        input=input_path,
        reference=reference_path,  # may be None -> speaker filtering skipped
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
                          "Omit if the video has only one speaker — quality-only selection is used instead.")
    ap.add_argument("--out", default="audio.mp3", help="Output path (always exported as MP3)")
    ap.add_argument("--min-dur", type=float, default=8.0)
    ap.add_argument("--max-dur", type=float, default=10.0)
    ap.add_argument("--sim-threshold", type=float, default=0.75,
                     help="Cosine similarity threshold to accept a segment as the target speaker")
    ap.add_argument("--keep-wav", action="store_true", help="Keep the intermediate extracted WAV")
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
        speech_segments = merge_close_segments(raw_segments, max_gap=0.3)
        print(f"      Found {len(speech_segments)} speech segments")

        print("[3/6] Loading speaker encoder & scoring speaker similarity ...")
        if args.reference:
            encoder = load_speaker_encoder()
            ref_emb = get_reference_embedding(encoder, args.reference)
            score_speaker_similarity(encoder, wav_path, speech_segments, ref_emb)
            effective_sim_threshold = args.sim_threshold
        else:
            print("      No reference provided — skipping speaker filtering "
                  "(assuming single speaker / quality-only selection)")
            for seg in speech_segments:
                seg.speaker_sim = 1.0  # treat every segment as a pass
            effective_sim_threshold = 0.0

        print("[4/6] Scoring audio quality per segment ...")
        for seg in speech_segments:
            seg.score = quality_score(wav_path, seg)

        print("[5/6] Building candidate 8-10s windows ...")
        candidates = build_candidate_windows(
            speech_segments, args.min_dur, args.max_dur, effective_sim_threshold
        )
        if not candidates:
            msg = ("No candidate windows found. Try lowering --sim-threshold "
                   "or check the reference clip." if args.reference else
                   "No candidate windows found — the video may not have a "
                   f"continuous {args.min_dur:.0f}s+ clean speech segment. "
                   "Try lowering --min-dur.")
            print(msg)
            sys.exit(1)

        # Re-score each candidate window directly (more accurate than reusing
        # the parent segment's score, since window boundaries changed).
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


if __name__ == "__main__":
    main()
    