#!/usr/bin/env python3
"""
auto_clip_pipeline.py

Fully automated pipeline that takes a video (mp4, mkv, etc.) and produces
a single high-quality 8-10s mono MP3 clip of a target speaker's voice,
suitable as a voice-cloning / TTS reference sample.

Pipeline stages
----------------
1. Extract audio          -> ffmpeg, 16kHz mono WAV
2. Voice Activity Detect  -> Silero-VAD, finds natural speech chunks, each
                              bounded by a real pause (never mid-word)
3. Window selection        -> combine contiguous VAD chunks into every
                              possible 8-10s window that starts on real
                              speech and ends on a real pause; falls back
                              to nearest-silence snapping only if one
                              unbroken stretch of speech has no pause at
                              all in range
4. Speaker isolation       -> resemblyzer speaker embedding similarity
                              against a reference clip, if provided
                              (skipped for single-speaker audio)
5. Quality scoring         -> SNR estimate + spectral-flatness (music/noise
                              rejection) + loudness sanity check
6. Export                  -> ffmpeg slice -> MP3/WAV

Usage
-----
Interactive mode (just run it, no flags):

    python auto_clip_pipeline.py

    It will scan the current directory and ./uploads/ for video and audio
    files and let you pick, or tell you where to `scp` a file in from your
    laptop, then ask for an output filename/format.

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
# Stage 5: window selection — natural start/end boundaries only
# --------------------------------------------------------------------------

def build_candidate_windows(atoms: List[Segment], min_dur: float, max_dur: float) -> List[Segment]:
    """
    Every item in `atoms` is a single VAD-detected speech chunk: it starts
    exactly where real speech begins and ends exactly where a real pause
    begins (VAD only splits at silence). To build an 8-10s clip that never
    starts or ends mid-word, we only ever combine a *contiguous run* of
    whole atoms — the run's start is atom[i]'s start (a real word start)
    and the run's end is atom[j]'s end (a real pause), with the small
    natural pauses between atoms simply included in the audio.

    Returns every such run whose total duration falls in [min_dur, max_dur].
    """
    atoms = sorted(atoms, key=lambda s: s.start)
    n = len(atoms)
    candidates: List[Segment] = []
    for i in range(n):
        for j in range(i, n):
            duration = atoms[j].end - atoms[i].start
            if duration > max_dur:
                break  # extending further only grows the duration
            if duration >= min_dur:
                candidates.append(Segment(atoms[i].start, atoms[j].end))
    return candidates


def snap_to_nearest_silence(
    full_wav_path: str, target_sample: int, search_radius_sec: float = 1.0, sr: int = 16000
) -> int:
    """
    Fallback for when no natural pause exists near the target cut point
    (e.g. one long unbroken stretch of talking longer than max_dur).
    Looks in a window around target_sample for the locally quietest
    moment (smallest short-time energy) and returns that sample index,
    so the cut lands on a breath/comma-pause instead of a random instant.
    """
    import soundfile as sf

    audio, file_sr = sf.read(full_wav_path)
    assert file_sr == sr
    radius = int(search_radius_sec * sr)
    lo = max(0, target_sample - radius)
    hi = min(len(audio), target_sample + radius)
    window = audio[lo:hi].astype(np.float32)
    if len(window) < 200:
        return target_sample

    frame_len = int(0.02 * sr)
    hop = int(0.01 * sr)
    best_idx, best_energy = target_sample, None
    for start in range(0, max(1, len(window) - frame_len), hop):
        frame = window[start:start + frame_len]
        energy = float(np.mean(frame ** 2))
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_idx = lo + start + frame_len // 2
    return best_idx


def build_fallback_window(
    atoms: List[Segment], full_wav_path: str, min_dur: float, max_dur: float, sr: int = 16000
) -> Optional[Segment]:
    """
    Used only when no atom (or combination of atoms) fits within
    [min_dur, max_dur] on its own — meaning some stretch of speech is
    longer than max_dur with no VAD-detected pause inside it. Picks the
    longest single atom >= min_dur and snaps its end to the nearest local
    silence within the target range, instead of an arbitrary time cut.
    """
    long_atoms = [a for a in atoms if a.duration >= min_dur]
    if not long_atoms:
        return None
    atom = max(long_atoms, key=lambda a: a.duration)
    target_sample = int((atom.start + max_dur) * sr)
    snapped_sample = snap_to_nearest_silence(full_wav_path, target_sample, sr=sr)
    snapped_end = max(atom.start + min_dur, min(snapped_sample / sr, atom.end))
    return Segment(atom.start, snapped_end)


# --------------------------------------------------------------------------
# Stage 6: export
# --------------------------------------------------------------------------

def export_audio(video_or_wav_path: str, seg: Segment, out_path: str, fmt: str = "mp3") -> None:
    codec_args = (
        ["-c:a", "libmp3lame", "-q:a", "2"] if fmt == "mp3"
        else ["-c:a", "pcm_s16le"]  # wav
    )
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

    fmt = ""
    while fmt not in ("mp3", "wav"):
        fmt = input("\nOutput format — mp3 or wav? [mp3]: ").strip().lower() or "mp3"

    default_name = f"clip.{fmt}"
    out_name = input(f"Output filename [{default_name}]: ").strip() or default_name
    if not out_name.lower().endswith(f".{fmt}"):
        out_name += f".{fmt}"

    ns = argparse.Namespace(
        input=input_path,
        reference=reference_path,  # may be None -> speaker filtering skipped
        out=out_name,
        format=fmt,
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
    ap.add_argument("--out", default="clip.mp3", help="Output path")
    ap.add_argument("--format", choices=["mp3", "wav"], default=None,
                     help="Output format; inferred from --out extension if omitted")
    ap.add_argument("--min-dur", type=float, default=8.0)
    ap.add_argument("--max-dur", type=float, default=10.0)
    ap.add_argument("--sim-threshold", type=float, default=0.75,
                     help="Cosine similarity threshold to accept a segment as the target speaker")
    ap.add_argument("--keep-wav", action="store_true", help="Keep the intermediate extracted WAV")
    args = ap.parse_args()

    if not args.input:
        args = interactive_config()
    elif args.format is None:
        args.format = "wav" if args.out.lower().endswith(".wav") else "mp3"

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "extracted.wav")
        print(f"[1/6] Extracting audio from {args.input} ...")
        extract_audio(args.input, wav_path)

        print("[2/6] Running VAD ...")
        atoms = run_vad(wav_path)
        print(f"      Found {len(atoms)} natural speech chunks "
              f"(each bounded by a real pause)")

        print("[3/6] Building candidate windows on natural pause boundaries ...")
        candidates = build_candidate_windows(atoms, args.min_dur, args.max_dur)
        used_fallback = False
        if not candidates:
            print("      No natural combination fits the range — falling back to "
                  "nearest-silence snapping for one long unbroken stretch of speech")
            fallback = build_fallback_window(atoms, wav_path, args.min_dur, args.max_dur)
            if fallback:
                candidates = [fallback]
                used_fallback = True

        if not candidates:
            print(f"No usable speech window found. The video may not have a "
                  f"continuous {args.min_dur:.0f}s+ clean speech stretch. "
                  f"Try lowering --min-dur.")
            sys.exit(1)

        print("[4/6] Scoring speaker similarity and audio quality per candidate ...")
        if args.reference:
            encoder = load_speaker_encoder()
            ref_emb = get_reference_embedding(encoder, args.reference)
            score_speaker_similarity(encoder, wav_path, candidates, ref_emb)
            candidates = [c for c in candidates if c.speaker_sim >= args.sim_threshold] or candidates
        else:
            print("      No reference provided — skipping speaker filtering "
                  "(assuming single speaker / quality-only selection)")

        for c in candidates:
            c.score = quality_score(wav_path, c)

        print("[5/6] Selecting the best-scoring window ...")
        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0]
        tag = " (silence-snapped fallback)" if used_fallback else ""
        print(f"      Best window: {best.start:.2f}s - {best.end:.2f}s{tag} "
              f"(duration {best.duration:.2f}s, quality score {best.score:.3f})")

        print(f"[6/6] Exporting to {args.out} ...")
        export_audio(args.input, best, args.out, fmt=args.format)

        if args.keep_wav:
            keep_path = os.path.splitext(args.out)[0] + "_full.wav"
            import shutil
            shutil.copy(wav_path, keep_path)
            print(f"      Kept intermediate WAV at {keep_path}")

    print("Done.")


if __name__ == "__main__":
    main()