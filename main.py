


import os
import io
import uuid
import time
import asyncio
import logging
import shutil
import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from cached_path import cached_path
from pydub import AudioSegment, silence

from f5_tts.api import F5TTS

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------
# FASTAPI
# ---------------------------------------------------------------------
app = FastAPI(title="Production F5-TTS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# LIMITS
# ---------------------------------------------------------------------
MAX_CONCURRENT_USERS = 20
USER_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_USERS)

VOICE_LOCKS: dict[str, asyncio.Lock] = {}

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
BASE_DIR = "/workspace/F5-TTS"
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(RESOURCES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------
# MODEL INIT (OFFICIAL)
# ---------------------------------------------------------------------
logging.info("Loading F5-TTS model...")

model = F5TTS(
    device=DEVICE,
    vocab_file=str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/vocab.txt")),
    ckpt_file=str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors")),
    vocoder_name="vocos",
    ode_method="euler",
    use_ema=True,
)


logging.info("F5-TTS model loaded")

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def get_voice_lock(voice: str) -> asyncio.Lock:
    if voice not in VOICE_LOCKS:
        VOICE_LOCKS[voice] = asyncio.Lock()
    return VOICE_LOCKS[voice]

def convert_to_wav(src: str, dst: str):
    audio = AudioSegment.from_file(src)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(24000)
    audio.export(dst, format="wav")

def trim_reference_audio(wav_path: str) -> str:
    audio = AudioSegment.from_file(wav_path)

    chunks = silence.split_on_silence(
        audio,
        min_silence_len=500,
        silence_thresh=-45,
        keep_silence=300,
    )

    clipped = AudioSegment.silent(duration=0)
    for c in chunks:
        if len(clipped) >= 14000:
            break
        clipped += c

    if len(clipped) == 0:
        clipped = audio[:14000]

    clipped = clipped[:15000] + AudioSegment.silent(duration=50)

    out_path = wav_path.replace(".wav", "_short.wav")
    clipped.export(out_path, format="wav")
    return out_path

def stream_wav(path: str):
    return StreamingResponse(open(path, "rb"), media_type="audio/wav")

# ---------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
    }

@app.get("/voices")
def voices():
    return [
        f.replace(".wav", "")
        for f in os.listdir(RESOURCES_DIR)
        if f.endswith(".wav")
    ]

@app.post("/upload_voice")
async def upload_voice(
    voice_name: str = Form(...),
    file: UploadFile = File(...)
):
    data = await file.read()
    ext = file.filename.split(".")[-1].lower()

    if ext not in {"wav", "mp3", "flac", "ogg"}:
        raise HTTPException(400, "Unsupported audio format")

    raw_path = os.path.join(RESOURCES_DIR, f"{voice_name}.{ext}")
    wav_path = os.path.join(RESOURCES_DIR, f"{voice_name}.wav")

    with open(raw_path, "wb") as f:
        f.write(data)

    convert_to_wav(raw_path, wav_path)
    os.remove(raw_path)

    return {"status": "voice uploaded", "voice": voice_name}

@app.post("/synthesize")
async def synthesize(
    text: str = Query(..., min_length=1),
    voice: str = Query("default_en"),
):
    await USER_SEMAPHORE.acquire()
    voice_lock = get_voice_lock(voice)

    request_id = uuid.uuid4().hex
    output_path = os.path.join(OUTPUT_DIR, f"{request_id}.wav")

    try:
        ref_path = os.path.join(RESOURCES_DIR, f"{voice}.wav")

        if not os.path.exists(ref_path):
            raise HTTPException(404, "Voice not found")

        async with voice_lock:
            short_ref = trim_reference_audio(ref_path)
            ref_text = model.transcribe(short_ref)

            model.infer(
                ref_file=short_ref,
                ref_text=ref_text,
                gen_text=text,
                speed=1.0,
                nfe_step=32,
                cfg_strength=2.0,
                file_wave=output_path,
            )

        return stream_wav(output_path)

    except Exception as e:
        logging.exception("Synthesis failed")
        raise HTTPException(500, str(e))

    finally:
        USER_SEMAPHORE.release()












# import io
# import asyncio
# import uuid
# import logging
# import torch
# import numpy as np
# import soundfile as sf

# from fastapi import FastAPI, UploadFile, File, Query, HTTPException
# from fastapi.responses import StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware

# # ---------------------------------------------------------------------
# # LOGGING (MANDATORY FOR PRODUCTION)
# # ---------------------------------------------------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
# )

# # ---------------------------------------------------------------------
# # FASTAPI INIT
# # ---------------------------------------------------------------------
# app = FastAPI(title="Production F5-TTS API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ---------------------------------------------------------------------
# # GLOBAL LIMITS
# # ---------------------------------------------------------------------
# MAX_CONCURRENT_USERS = 20
# USER_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_USERS)

# WORDS_PER_CHUNK = 5
# DEFAULT_SAMPLE_RATE = 22050

# # ---------------------------------------------------------------------
# # MODEL REGISTRY
# # ---------------------------------------------------------------------
# MODEL_CONFIGS = {
#     "f5_default": {
#         "weights": "weights/f5_tts_base.pt",
#     }
# }

# TTS_MODELS = {}
# VOICE_QUEUES = {}

# # ---------------------------------------------------------------------
# # LOAD MODELS (SAFE)
# # ---------------------------------------------------------------------
# def load_model(weights_path: str):
#     try:
#         model = torch.load(weights_path, map_location="cuda")
#         model.eval()
#         return model
#     except Exception as e:
#         logging.critical(f"Failed to load model: {e}")
#         raise RuntimeError("Model load failure")

# for key, cfg in MODEL_CONFIGS.items():
#     TTS_MODELS[key] = load_model(cfg["weights"])
#     VOICE_QUEUES[key] = asyncio.Queue()

# # ---------------------------------------------------------------------
# # AUDIO HELPERS
# # ---------------------------------------------------------------------
# def chunk_text(text: str):
#     words = text.strip().split()
#     return [" ".join(words[i:i+WORDS_PER_CHUNK]) for i in range(0, len(words), WORDS_PER_CHUNK)]

# def wav_to_buffer(wav: np.ndarray, sr: int):
#     buf = io.BytesIO()
#     sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
#     buf.seek(0)
#     return buf

# def concat_audio(chunks):
#     return np.concatenate(chunks, axis=0)

# # ---------------------------------------------------------------------
# # VOICE WORKER (SEQUENTIAL PER VOICE)
# # ---------------------------------------------------------------------
# async def voice_worker(model_key: str):
#     model = TTS_MODELS[model_key]
#     queue = VOICE_QUEUES[model_key]

#     while True:
#         job = await queue.get()
#         try:
#             future, payload = job
#             result = await synthesize_internal(model, **payload)
#             future.set_result(result)
#         except Exception as e:
#             future.set_exception(e)
#         finally:
#             queue.task_done()

# # ---------------------------------------------------------------------
# # INTERNAL SYNTHESIS
# # ---------------------------------------------------------------------
# async def synthesize_internal(model, text, ref_wav, ref_sr):
#     audio_chunks = []

#     for chunk in chunk_text(text):
#         with torch.inference_mode():
#             wav, sr = model.infer(chunk, ref_wav, ref_sr)
#             audio_chunks.append(wav)

#         torch.cuda.empty_cache()

#     final_audio = concat_audio(audio_chunks)
#     return wav_to_buffer(final_audio, sr)

# # ---------------------------------------------------------------------
# # START WORKERS
# # ---------------------------------------------------------------------
# @app.on_event("startup")
# async def startup():
#     for key in VOICE_QUEUES:
#         asyncio.create_task(voice_worker(key))
#     logging.info("Voice workers started")

# # ---------------------------------------------------------------------
# # API ENDPOINTS
# # ---------------------------------------------------------------------
# @app.get("/health")
# def health():
#     return {"status": "ok", "models": list(TTS_MODELS.keys())}

# @app.get("/voices")
# def voices():
#     return list(TTS_MODELS.keys())

# @app.post("/synthesize")
# async def synthesize(
#     text: str = Query(..., min_length=1),
#     model_key: str = Query("f5_default"),
#     ref_audio: UploadFile = File(...)
# ):
#     if model_key not in TTS_MODELS:
#         raise HTTPException(404, "Model not found")

#     await USER_SEMAPHORE.acquire()

#     try:
#         ref_bytes = await ref_audio.read()
#         ref_wav, ref_sr = sf.read(io.BytesIO(ref_bytes), dtype="float32")

#         if ref_sr <= 0 or ref_wav.size == 0:
#             raise ValueError("Invalid reference audio")

#         future = asyncio.get_event_loop().create_future()

#         await VOICE_QUEUES[model_key].put(
#             (future, {
#                 "text": text,
#                 "ref_wav": ref_wav,
#                 "ref_sr": ref_sr
#             })
#         )

#         buffer = await future
#         return StreamingResponse(buffer, media_type="audio/wav")

#     except Exception as e:
#         logging.error(f"Synthesis failed: {e}")
#         raise HTTPException(500, "TTS generation failed")

#     finally:
#         USER_SEMAPHORE.release()
