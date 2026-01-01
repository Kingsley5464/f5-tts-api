

import io
import os
import asyncio
import logging
import torch
import numpy as np
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from f5_tts.api import F5TTS
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------
# APP INIT
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
MAX_CONCURRENT_USERS = 8
USER_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_USERS)

WORDS_PER_CHUNK = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------
# MODEL REGISTRY
# ---------------------------------------------------------------------
MODEL_CONFIGS = {
    "f5_base": {
        "repo": "SWivid/F5-TTS",
        "ckpt": "F5TTS_Base/model_1200000.safetensors",
        "vocab": "F5TTS_Base/vocab.txt",
    }
}

TTS_MODELS = {}
MODEL_QUEUES = {}

# ---------------------------------------------------------------------
# DEFAULT VOICES (REFERENCE AUDIO)
# ---------------------------------------------------------------------
DEFAULT_VOICES = {
    "neutral": "voices/neutral.wav",
    "female": "voices/female.wav",
    "male": "voices/male.wav",
}

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def chunk_text(text: str):
    words = text.strip().split()
    return [
        " ".join(words[i:i + WORDS_PER_CHUNK])
        for i in range(0, len(words), WORDS_PER_CHUNK)
    ]


def wav_to_buffer(wav: np.ndarray, sr: int):
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


def concat_audio(chunks):
    return np.concatenate(chunks, axis=0)


def load_wav(path: str):
    wav, sr = sf.read(path, dtype="float32")
    return wav, sr


# ---------------------------------------------------------------------
# LOAD MODELS (SAFE)
# ---------------------------------------------------------------------
def load_model(cfg):
    logging.info("Loading F5-TTS model...")
    vocab = hf_hub_download(cfg["repo"], cfg["vocab"])
    ckpt = hf_hub_download(cfg["repo"], cfg["ckpt"])

    model = F5TTS(
        device=DEVICE,
        vocab_file=vocab,
        ckpt_file=ckpt,
    )

    return model


for key, cfg in MODEL_CONFIGS.items():
    TTS_MODELS[key] = load_model(cfg)
    MODEL_QUEUES[key] = asyncio.Queue()

# ---------------------------------------------------------------------
# WORKER (GPU SAFE)
# ---------------------------------------------------------------------
async def model_worker(model_key: str):
    model = TTS_MODELS[model_key]
    queue = MODEL_QUEUES[model_key]

    while True:
        future, payload = await queue.get()
        try:
            result = await synthesize_internal(model, **payload)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            queue.task_done()


# ---------------------------------------------------------------------
# CORE SYNTHESIS
# ---------------------------------------------------------------------
async def synthesize_internal(model, text, ref_wav, ref_sr):
    audio_chunks = []

    for chunk in chunk_text(text):
        with torch.inference_mode():
            # NOTE:
            # This matches CURRENT F5-TTS inference API.
            # If your version differs, this is the ONLY line to adjust.
            wav, sr = model.infer(chunk, ref_wav, ref_sr)

        audio_chunks.append(wav)

    final_audio = concat_audio(audio_chunks)
    return wav_to_buffer(final_audio, sr)


# ---------------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    for key in MODEL_QUEUES:
        asyncio.create_task(model_worker(key))
    logging.info("TTS workers started")


# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": list(TTS_MODELS.keys()),
        "voices": list(DEFAULT_VOICES.keys()),
        "device": DEVICE,
    }


@app.get("/voices")
def voices():
    return list(DEFAULT_VOICES.keys())


@app.post("/synthesize")
async def synthesize(
    text: str = Query(..., min_length=1),
    model_key: str = Query("f5_base"),
    voice: str = Query("neutral"),
    ref_audio: UploadFile | None = File(None),
):
    if model_key not in TTS_MODELS:
        raise HTTPException(404, "Model not found")

    await USER_SEMAPHORE.acquire()

    try:
        # -------------------------------
        # Reference audio selection
        # -------------------------------
        if ref_audio:
            ref_bytes = await ref_audio.read()
            ref_wav, ref_sr = sf.read(io.BytesIO(ref_bytes), dtype="float32")
        else:
            if voice not in DEFAULT_VOICES:
                raise HTTPException(400, "Invalid voice")
            ref_wav, ref_sr = load_wav(DEFAULT_VOICES[voice])

        if ref_wav.size == 0 or ref_sr <= 0:
            raise ValueError("Invalid reference audio")

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        await MODEL_QUEUES[model_key].put(
            (future, {
                "text": text,
                "ref_wav": ref_wav,
                "ref_sr": ref_sr,
            })
        )

        buffer = await future
        return StreamingResponse(buffer, media_type="audio/wav")

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
