"""FastAPI service.  uvicorn app.api:app --host 127.0.0.1 --port 8000   (one worker: there is one GPU)"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .documents import DocumentError, extract
from .engine import ENGINE, MAX_WORDS, InputError

log = logging.getLogger("factindic")
Kind = Literal["Auto", "Hindi", "Gujarati", "Hinglish", "Gujlish", "English"]
Mode = Literal["full", "fast"]
MAX_BYTES, MAX_CHARS = 10_000_000, 400_000


def _load():
    try:
        ENGINE.load()
    except Exception:
        log.exception("model loading failed")


@asynccontextmanager
async def lifespan(_):
    threading.Thread(target=_load, daemon=True).start()          # /health answers while the models load
    yield
    if getattr(ENGINE, "mt", None):
        ENGINE.mt.t.close()


app = FastAPI(title="FactIndic", description="Factuality-aware summaries of Hindi, Gujarati, Hinglish, Gujlish and English documents",
              version="1.0", lifespan=lifespan)


class Request(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    language: Kind = "Auto"
    mode: Mode = "full"
    words: int = Field(40, ge=25, le=100, description="target length of the summary")


def _run(text, language, mode, words):
    if not ENGINE.ready:
        raise HTTPException(503, ENGINE.error or f"The models are not ready yet ({ENGINE.stage}).")
    try:
        return ENGINE.summarize(text, language, mode, words)
    except InputError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        log.exception("summarization failed")
        if "out of memory" in str(e).lower():
            raise HTTPException(507, "The GPU ran out of memory. Use a shorter document or the fast mode.")
        raise HTTPException(500, f"Summarization failed ({type(e).__name__}).")


@app.get("/health")
def health():
    return dict(ready=ENGINE.ready, stage=ENGINE.stage, error=ENGINE.error, max_words=MAX_WORDS)


@app.post("/summarize")
def summarize(r: Request):
    return _run(r.text, r.language, r.mode, r.words)


@app.post("/summarize/file")
def summarize_file(file: UploadFile = File(...), language: Kind = Form("Auto"), mode: Mode = Form("full"),
                   words: int = Form(40, ge=25, le=100)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The file is larger than 10 MB.")
    try:
        text = extract(file.filename, data)
    except DocumentError as e:
        raise HTTPException(422, str(e))
    if not text:
        raise HTTPException(422, "The file contains no text.")
    return _run(text[:MAX_CHARS], language, mode, words)
