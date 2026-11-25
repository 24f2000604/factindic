# FactIndic demo (FastAPI + uvicorn + Streamlit)

```bash
./run_app.sh          # UI http://localhost:8501 · API docs http://127.0.0.1:8000/docs · Ctrl-C stops both
```
Needs the project environment (`setup_local.sh`), the downloaded models, `runs/factindic/data` and an 8 GB NVIDIA GPU; nothing else may
use the GPU. Both services listen on this computer only and have no login: do not expose them as they are. Documents are not stored.

| Input | Read as | Summary in |
|---|---|---|
| Hindi, Gujarati | native script | Hindi, Gujarati |
| Hinglish, Gujlish | Romanized or script-mixed (detected) | Hindi (Devanagari), Gujarati script |
| English | native (extension, **not evaluated**) | English |

- **full**: the proposed model. Anchors → B1 prompt → Anchor-CAD + DoLa candidate and 4 samples → round-trip router → J reranking → prune. 9–20 s.
- **fast**: B1 prompt only, no fact check, 1.5–4 s.
- The first 1,000 words are read. The experiments used articles cut to 350 words.
- For Romanized input the anchors come from back-transliterated text; the experiments had the aligned native article, so expect lower quality.

```bash
curl -s localhost:8000/summarize -H 'content-type: application/json' -d '{"text":"…","language":"Auto","mode":"full","words":40}'
curl -s localhost:8000/summarize/file -F file=@article.pdf -F language=Auto -F mode=full -F words=40
```
Files: `engine.py` pipeline and GPU memory · `language.py` input detection · `documents.py` PDF, DOCX, TXT · `api.py` · `ui.py`.
