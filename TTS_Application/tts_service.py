"""
tts_service.py -- dubbing service: video -> speech in the target language,
in the voice of the speaker in the video.

Flow of one job
  1. Cut a clean 8-10 s reference clip of the speaker from the video.
  2. Transcribe that clip in the source language          -> ref_text
  3. Translate ref_text into the target language          -> translated_text
  4. IndicF5 speaks translated_text in the speaker's voice
     (reference = the clip from step 1 + ref_text)
  5. Result = ONE wav file (24 kHz)

Run:
  TTS_API_KEY=<key> uvicorn tts_service:app --host 0.0.0.0 --port 8000
  (do NOT use --workers > 1)

Endpoints
  POST /tts           multipart: video (file), reference (file, optional),
                      source_lang (hi|kn|te|ta|en), target_lang (hi|kn|te|ta)
                      -> {"job_id": "..."}
  GET  /status/{id}   -> status queued|running|done|failed,
                         plus ref_text and translated_text when done
  GET  /result/{id}   -> dubbed.wav
  GET  /health
All endpoints except /health need the header  x-api-key: <key>
"""

import json
import os
import shutil
import tempfile
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

# Importing this loads IndicF5 onto the device once, at server start.
import merged_pipeline2 as core

API_KEY = os.environ["TTS_API_KEY"]
JOBS_DIR = os.path.abspath("jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

TARGET_LANGS = {"hi", "kn", "te", "ta"}
SOURCE_LANGS = TARGET_LANGS | {"en"}


class UTF8JSONResponse(JSONResponse):
    """Show Hindi/Kannada/... text as real characters instead of \\uXXXX escapes."""
    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def extract_reference_clip(input_path, reference_path, out_path,
                           min_dur=8.0, max_dur=10.0, sim_threshold=0.75):
    """Stage 1: video -> best 8-10 s clean clip of the target speaker (mp3)."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "extracted.wav")
        core.extract_audio(input_path, wav)

        segs = core.merge_close_segments(core.run_vad(wav), max_gap=0.3)

        if reference_path:
            enc = core.load_speaker_encoder()
            ref_emb = core.get_reference_embedding(enc, reference_path)
            core.score_speaker_similarity(enc, wav, segs, ref_emb)
            thr = sim_threshold
        else:
            for s in segs:
                s.speaker_sim = 1.0
            thr = 0.0

        for s in segs:
            s.score = core.quality_score(wav, s)

        cands = core.build_candidate_windows(segs, min_dur, max_dur, thr)
        if not cands:
            raise ValueError(
                "No candidate window found (no continuous clean speech of the "
                "target speaker long enough, or reference clip doesn't match)."
            )
        for c in cands:
            c.score = core.quality_score(wav, c)
        best = max(cands, key=lambda c: c.score)
        core.export_audio(input_path, best, out_path)


def dub(ref_audio_path, source_lang, target_lang, out_wav):
    """Stages 2-4: transcribe -> translate -> speak the translation in the speaker's voice."""
    # Reference text = transcript in the SAME language as the reference audio.
    ref_text = core.get_ref_text(ref_audio_path, source_lang, source_lang)

    # Text to speak = that transcript translated into the target language.
    translated_text = core.translate_with_indictrans2(ref_text, source_lang, target_lang)
    print(f"\nText to speak ({target_lang}): {translated_text}\n")

    audio = core.model(translated_text, ref_audio_path=ref_audio_path, ref_text=ref_text)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    sf.write(out_wav, np.array(audio, dtype=np.float32), samplerate=24000)
    return ref_text, translated_text


# ---------------------------------------------------------------------------
# Job machinery
# ---------------------------------------------------------------------------

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=1)  # one job at a time; others queue
jobs = {}  # job_id -> dict (in memory; lost if the server restarts)


def check_key(key: Optional[str]):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def process(job_id, video_path, ref_path, source_lang, target_lang):
    d = os.path.join(JOBS_DIR, job_id)
    try:
        jobs[job_id]["status"] = "running"

        clip = os.path.join(d, "audio.mp3")
        extract_reference_clip(video_path, ref_path, clip)

        out_wav = os.path.join(d, "dubbed.wav")
        ref_text, translated_text = dub(clip, source_lang, target_lang, out_wav)

        jobs[job_id].update(status="done", result=out_wav,
                            ref_text=ref_text, translated_text=translated_text)
    except Exception as e:
        traceback.print_exc()
        jobs[job_id].update(status="failed", error=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/tts")
async def submit(
    video: UploadFile = File(...),
    reference: Optional[UploadFile] = File(None),
    source_lang: str = Form(...),
    target_lang: str = Form(...),
    x_api_key: Optional[str] = Header(None),
):
    check_key(x_api_key)
    if source_lang not in SOURCE_LANGS or target_lang not in TARGET_LANGS:
        raise HTTPException(400, f"source_lang in {sorted(SOURCE_LANGS)}, "
                                 f"target_lang in {sorted(TARGET_LANGS)}")

    job_id = uuid.uuid4().hex
    d = os.path.join(JOBS_DIR, job_id)
    os.makedirs(d)

    video_path = os.path.join(d, "input" + os.path.splitext(video.filename or ".mp4")[1])
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    ref_path = None
    if reference is not None and reference.filename:
        ref_path = os.path.join(d, "reference" + os.path.splitext(reference.filename)[1])
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)

    jobs[job_id] = {"status": "queued", "source_lang": source_lang, "target_lang": target_lang}
    executor.submit(process, job_id, video_path, ref_path, source_lang, target_lang)
    return {"job_id": job_id}


@app.get("/status/{job_id}", response_class=UTF8JSONResponse)
def status(job_id: str, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    return {k: v for k, v in job.items() if k != "result"}


@app.get("/result/{job_id}")
def result(job_id: str, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    return FileResponse(job["result"], media_type="audio/wav",
                        filename=f"{job_id}_dubbed.wav")