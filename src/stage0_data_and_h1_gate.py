# %% [markdown]
# # FactIndic — Experiment 0: the H1 Gate (Tokenizer Fertility vs. Claim-Level Hallucination)
#
# **Run on Kaggle (GPU T4, Internet ON) or Colab (T4).** Nothing here is meant to run on a local CPU.
#
# H1: at matched content and summary length, higher subword fertility F = T(D)/W(D) is positively
# associated with claim-level hallucination rate, (a) within one model across scripts (native vs
# Romanized) and (b) across model families (Sarvam-1 vs Gemma-4 vs Llama-3.2).
#
# Gate: keep `sarvamai/sarvam-1` (+LoRA SFT) as the Stage 2–3 backbone only if H1 holds **and**
# Sarvam-1-SFT is competitive on hallucination **and** its generation quality is acceptable;
# otherwise switch to Gemma-4.
#
# Phases (each one resumable; outputs go to `OUT/` as JSONL and are skipped if already present):
# 1. Data: XL-Sum hi/gu → 100-article pilot (50+50), 1,000+1,000 fertility sentences, 3,000 SFT pairs
# 2. Input variants: native / script-mixed / Romanized
# 3. Fertility table (Sarvam-1, Gemma-4, Llama-3.2, IndicBARTSS)
# 4. LoRA SFT on Sarvam-1 (r=16, α=32, q/k/v/o → 6.42M params)
# 5. Greedy generation: 3 models × {native, Romanized} × 100 articles
# 6. Grading: claim-level (IndicTrans2→MiniCheck-FT5, native mDeBERTa-XNLI, round-trip router), anchors, ROUGE
# 7. Statistics, gate decision, report
#
# **Setup before running:** add a Kaggle Secret (or Colab secret) named `HF_TOKEN`, and accept the
# licences on Hugging Face for `meta-llama/Llama-3.2-3B-Instruct`, `google/gemma-4-E2B-it` and
# `sarvamai/sarvam-1` with the same account.

# %%
import os
from huggingface_hub import login, HfApi

# Load token from Kaggle or Colab secrets
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except ImportError:
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")

os.environ["HF_TOKEN"] = hf_token
login(token=hf_token)

# Verify access to all 3 gated models
api = HfApi()
for repo in ["meta-llama/Llama-3.2-3B-Instruct", "google/gemma-4-E2B-it", "sarvamai/sarvam-1"]:
    try:
        api.model_info(repo)
        print(f"Access verified: {repo}")
    except Exception as e:
        print(f"Access failed for {repo}: {e}")

# %%
# !pip install -q "accelerate>=1.5.0" "datasets>=3.3.0" "peft>=0.14.0" "numpy>=2.0.0,<2.3.0" "pyarrow>=19.0.0"
# !pip install -q --no-deps --force-reinstall "transformers==5.24.0" "huggingface_hub==1.33.0"

# %%
import transformers, huggingface_hub
print("transformers:", transformers.__version__)
print("huggingface_hub:", huggingface_hub.__version__)

# %% Install
import os, sys, subprocess

def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

_pip("-U", "transformers", "accelerate", "peft", "bitsandbytes", "datasets", "huggingface_hub")
_pip("sentencepiece", "protobuf", "indic-nlp-library", "indic-transliteration", "rouge-score",
     "statsmodels", "scipy", "pandas", "rapidfuzz", "tabulate", "matplotlib", "safetensors")
try:
    _pip("IndicTransToolkit")
except Exception as e:  # falls back to NLLB-200 at grading time
    print("IndicTransToolkit install failed, will fall back to NLLB:", e)

# %% Config, paths, auth
import gc, json, math, random, re, time, hashlib, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import torch

IS_KAGGLE = os.path.exists("/kaggle/working")
IN_COLAB = "google.colab" in sys.modules

if IS_KAGGLE:
    OUT = Path("/kaggle/working/factindic_h1")
elif IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = Path("/content/drive/MyDrive/factindic_h1")  # survives Colab disconnects
else:
    OUT = Path("./factindic_h1")
OUT.mkdir(parents=True, exist_ok=True)
if IS_KAGGLE and not any(OUT.iterdir()):
    # Resume: attach a previous run's output (Add Input → your notebook / dataset) and it is copied here.
    import glob, shutil
    prev = [p for p in glob.glob("/kaggle/input/**/factindic_h1", recursive=True) if os.path.isdir(p)]
    if prev:
        shutil.copytree(prev[0], OUT, dirs_exist_ok=True)
        print("restored previous outputs from", prev[0])

def _get_hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return None

HF_TOKEN = _get_hf_token()
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    os.environ["HF_TOKEN"] = HF_TOKEN
else:
    print("WARNING: no HF_TOKEN found; gated models (Llama-3.2, maybe Gemma/Sarvam) will fail.")

CFG = dict(
    seed=42,
    n_fert_sent=1000,          # per language
    n_pilot_per_lang=50,
    n_sft_per_lang=1500,       # half get Romanized inputs
    max_article_words=350,     # same truncated D for every model and every script variant
    sft_max_len=1024,
    sft_micro_bs=2, sft_accum=8, sft_epochs=1, sft_lr=2e-4,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    tau_ver=0.5,
    max_new_tokens=192,
    models={
        "sarvam1_sft": dict(hf="sarvamai/sarvam-1", kind="base_sft"),
        "gemma4_e2b": dict(hf="google/gemma-4-E2B-it", kind="chat"),
        "llama32_3b": dict(hf="meta-llama/Llama-3.2-3B-Instruct", kind="chat"),
    },
    fert_only={"indicbartss": "ai4bharat/IndicBARTSS"},
    mt="ai4bharat/indictrans2-indic-en-dist-200M",
    mt_fallback="facebook/nllb-200-distilled-600M",
    minicheck="lytang/MiniCheck-Flan-T5-Large",
    nli="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    ner="ai4bharat/IndicNER",
)
json.dump(CFG, open(OUT / "config.json", "w"), indent=2)

random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
DEV = "cuda" if torch.cuda.is_available() else "cpu"
assert DEV == "cuda", "Enable the T4 GPU accelerator (Kaggle: Settings → Accelerator → GPU T4)."
print(torch.cuda.get_device_name(0), "| OUT =", OUT)

LANGS = {"hi": dict(name="Hindi", script="Devanagari", xlsum="hindi", it2="hin_Deva",
                    block=(0x0900, 0x097F)),
         "gu": dict(name="Gujarati", script="Gujarati", xlsum="gujarati", it2="guj_Gujr",
                    block=(0x0A80, 0x0AFF))}

# %% Utilities: JSONL checkpointing, text helpers
def jl_write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def jl_append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()

def jl_read(path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a half-written last line after a disconnect
    return out

def free_gpu():
    """Call after `del model` in the caller; a callee cannot drop the caller's references."""
    gc.collect()
    torch.cuda.empty_cache()

# Explicit punctuation set: Python's \W also matches Indic vowel signs, so it must not be used to strip.
PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") | set("।॥“”‘’…–—₹«»·•")

def strip_punct(tok):
    i, j = 0, len(tok)
    while i < j and tok[i] in PUNCT:
        i += 1
    while j > i and tok[j - 1] in PUNCT:
        j -= 1
    return tok[:i], tok[i:j], tok[j:]

def words(text):
    """Orthographic words: whitespace tokens that are not pure punctuation."""
    return [w for w in text.split() if strip_punct(w)[1]]

def W(text):
    return len(words(text))

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
_NORM = {l: IndicNormalizerFactory().get_normalizer(l) for l in LANGS}

def clean(text, lang):
    text = unicodedata.normalize("NFC", text)
    text = _NORM[lang].normalize(text)
    return re.sub(r"\s+", " ", text).strip()

def truncate_words(text, n):
    toks = text.split()
    return " ".join(toks[:n])

def split_sentences(text):
    parts = re.split(r"(?<=[।.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def script_fraction(text, lang):
    lo, hi = LANGS[lang]["block"]
    letters = [c for c in text if c.isalpha() or unicodedata.category(c).startswith("M")]
    if not letters:
        return 0.0
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters)

# %% Romanization (casual Hinglish/Gujlish style) and script-mixing
# IndicXlit pulls in fairseq, which fails to install on current Kaggle/Colab Python. This uses
# rule-based ITRANS with schwa deletion and long-vowel collapsing instead, which approximates casual
# chat Romanization. Like real Hinglish it loses information (A/a, retroflex/dental merge).
from indic_transliteration import sanscript

_SCHEME = {"hi": sanscript.DEVANAGARI, "gu": sanscript.GUJARATI}
_INDIC_CHAR = re.compile(r"[ऀ-ॿ઀-૿]")
_CONS = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"

def _casual(word_itrans):
    s = word_itrans
    for a, b in [("R^i", "ri"), ("R^I", "ri"), ("L^i", "li"), (".N", "n"), ("~N", "n"), ("~n", "n"),
                 (".h", ""), (".a", ""), ("{}", ""), (".c", ""), ("^", ""), ("~", "n")]:
        s = s.replace(a, b)
    s = s.replace("|", ".")
    # final-schwa deletion: inherent short 'a' after a consonant at word end (long A stays)
    if len(s) > 3 and s.endswith("a") and s[-2] in _CONS:
        s = s[:-1]
    s = s.replace("M", "n").replace("H", "h").replace("x", "ksh")
    s = s.lower()
    s = re.sub(r"aa+", "a", s); s = re.sub(r"ii+", "i", s); s = re.sub(r"uu+", "u", s)
    s = re.sub(r"ee+", "e", s); s = re.sub(r"oo+", "o", s)
    return s

def romanize(text, lang):
    out = []
    for tok in text.split():
        pre, core, post = strip_punct(tok)
        if _INDIC_CHAR.search(core):
            core = _casual(sanscript.transliterate(core, _SCHEME[lang], sanscript.ITRANS))
        out.append((pre + core + post).replace("।", ".").replace("॥", "."))
    return " ".join(out)

GU_SUFFIX = sorted(["માં", "થી", "ને", "નો", "ની", "નું", "ના", "નાં", "એ"], key=len, reverse=True)

LEX_RAW = {
 "hi": {"सरकार": "government", "प्रदूषण": "pollution", "स्तर": "level", "चुनाव": "election",
        "अस्पताल": "hospital", "मंत्री": "minister", "प्रधानमंत्री": "prime minister",
        "मुख्यमंत्री": "chief minister", "अदालत": "court", "न्यायालय": "court",
        "विश्वविद्यालय": "university", "छात्र": "student", "छात्रों": "students", "शिक्षा": "education",
        "स्वास्थ्य": "health", "बाज़ार": "market", "बाजार": "market", "कंपनी": "company",
        "रिपोर्ट": "report", "फ़ैसला": "decision", "फैसला": "decision", "फ़ैसले": "decision",
        "फैसले": "decision", "सुरक्षा": "security", "कानून": "law", "क़ानून": "law", "नीति": "policy",
        "अर्थव्यवस्था": "economy", "विकास": "development", "परियोजना": "project", "समस्या": "problem",
        "जानकारी": "information", "मामला": "case", "मामले": "case", "जांच": "investigation",
        "जाँच": "investigation", "घटना": "incident", "दुर्घटना": "accident", "मौत": "death",
        "देश": "country", "शहर": "city", "गांव": "village", "गाँव": "village", "बिजली": "electricity",
        "सड़क": "road", "किसान": "farmer", "किसानों": "farmers", "युद्ध": "war", "सेना": "army",
        "हमला": "attack", "हमले": "attack", "विरोध": "protest", "प्रदर्शन": "protest",
        "बयान": "statement", "अधिकारी": "officer", "अधिकारियों": "officers", "कर्मचारी": "employee",
        "कर्मचारियों": "employees", "नौकरी": "job", "वैज्ञानिक": "scientist", "शोध": "research",
        "अध्ययन": "study", "तकनीक": "technology", "फ़िल्म": "film", "फिल्म": "film",
        "खिलाड़ी": "player", "मैच": "match", "टीम": "team", "जलवायु": "climate", "तापमान": "temperature",
        "बारिश": "rain", "बाढ़": "flood", "वायरस": "virus", "टीका": "vaccine", "वैक्सीन": "vaccine",
        "मरीज़": "patient", "मरीज": "patient", "मरीज़ों": "patients", "मरीजों": "patients",
        "डॉक्टर": "doctor", "इलाज": "treatment", "संक्रमण": "infection", "आंकड़े": "data",
        "आंकड़ों": "data", "सदस्य": "member", "संसद": "parliament", "पार्टी": "party", "नेता": "leader",
        "उम्मीदवार": "candidate", "वोट": "vote", "मतदान": "voting", "पुलिस": "police"},
 "gu": {"સરકાર": "government", "પ્રદૂષણ": "pollution", "સ્તર": "level", "ચૂંટણી": "election",
        "હોસ્પિટલ": "hospital", "મંત્રી": "minister", "વડાપ્રધાન": "prime minister",
        "મુખ્યમંત્રી": "chief minister", "કોર્ટ": "court", "અદાલત": "court",
        "યુનિવર્સિટી": "university", "વિદ્યાર્થી": "student", "વિદ્યાર્થીઓ": "students",
        "શિક્ષણ": "education", "આરોગ્ય": "health", "બજાર": "market", "કંપની": "company",
        "રિપોર્ટ": "report", "નિર્ણય": "decision", "સુરક્ષા": "security", "કાયદો": "law",
        "કાયદા": "law", "નીતિ": "policy", "અર્થતંત્ર": "economy", "વિકાસ": "development",
        "પ્રોજેક્ટ": "project", "સમસ્યા": "problem", "માહિતી": "information", "કેસ": "case",
        "તપાસ": "investigation", "ઘટના": "incident", "અકસ્માત": "accident", "મૃત્યુ": "death",
        "દેશ": "country", "શહેર": "city", "ગામ": "village", "વીજળી": "electricity", "રસ્તો": "road",
        "ખેડૂત": "farmer", "ખેડૂતો": "farmers", "યુદ્ધ": "war", "સેના": "army", "હુમલો": "attack",
        "વિરોધ": "protest", "પ્રદર્શન": "protest", "નિવેદન": "statement", "અધિકારી": "officer",
        "અધિકારીઓ": "officers", "કર્મચારી": "employee", "કર્મચારીઓ": "employees", "નોકરી": "job",
        "વૈજ્ઞાનિક": "scientist", "સંશોધન": "research", "અભ્યાસ": "study", "ટેકનોલોજી": "technology",
        "ફિલ્મ": "film", "ખેલાડી": "player", "મેચ": "match", "ટીમ": "team", "તાપમાન": "temperature",
        "વરસાદ": "rain", "પૂર": "flood", "વાયરસ": "virus", "રસી": "vaccine", "દર્દી": "patient",
        "દર્દીઓ": "patients", "ડૉક્ટર": "doctor", "સારવાર": "treatment", "ચેપ": "infection",
        "આંકડા": "data", "સભ્ય": "member", "સંસદ": "parliament", "પક્ષ": "party", "પાર્ટી": "party",
        "નેતા": "leader", "ઉમેદવાર": "candidate", "મતદાન": "voting", "પોલીસ": "police"},
}
LEX = {l: {clean(k, l): v for k, v in d.items()} for l, d in LEX_RAW.items()}

def script_mix(text, lang):
    """Swap common content nouns for Latin-script English inside the native matrix frame."""
    lex, out = LEX[lang], []
    for tok in text.split():
        pre, core, post = strip_punct(tok)
        rep = lex.get(core)
        if rep is None and lang == "gu":
            for suf in GU_SUFFIX:
                if core.endswith(suf) and core[: -len(suf)] in lex:
                    rep = lex[core[: -len(suf)]] + " " + suf  # "pollution નું"
                    break
        out.append(pre + rep + post if rep else tok)
    return " ".join(out)

def cmi_by_script(text):
    """Gambäck & Das CMI using script as the language tag; only valid for the script-mixed variant."""
    n = u = en = ind = 0
    for w in text.split():
        core = strip_punct(w)[1]
        n += 1
        if not core or core.replace(".", "").replace(",", "").isdigit():
            u += 1
        elif _INDIC_CHAR.search(core):
            ind += 1
        elif re.search(r"[A-Za-z]", core):
            en += 1
        else:
            u += 1
    return 0.0 if n <= u else 100 * (1 - max(en, ind) / (n - u))

# %% Normalizer A-5 (numbers) and cross-script entity keys
DIGITS = str.maketrans("०१२३४५६७८९૦૧૨૩૪૫૬૭૮૯", "01234567890123456789")
_nfc = lambda s: unicodedata.normalize("NFC", s)
MULT = {_nfc(k): v for k, v in {
    "सौ": 1e2, "સો": 1e2, "hundred": 1e2, "हज़ार": 1e3, "हजार": 1e3, "હજાર": 1e3, "hazar": 1e3,
    "thousand": 1e3, "लाख": 1e5, "લાખ": 1e5, "lakh": 1e5, "lakhs": 1e5, "करोड़": 1e7, "करोड": 1e7,
    "કરોડ": 1e7, "crore": 1e7, "crores": 1e7, "million": 1e6, "मिलियन": 1e6, "મિલિયન": 1e6,
    "billion": 1e9, "अरब": 1e9, "અબજ": 1e9, "बिलियन": 1e9, "બિલિયન": 1e9}.items()}
WORDNUM = {_nfc(k): v for k, v in {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9,
    "दस": 10, "बीस": 20, "पचास": 50, "डेढ़": 1.5, "ढाई": 2.5, "સવા": 1.25, "सवा": 1.25,
    "એક": 1, "બે": 2, "ત્રણ": 3, "ચાર": 4, "પાંચ": 5, "છ": 6, "સાત": 7, "આઠ": 8, "નવ": 9, "દસ": 10,
    "વીસ": 20, "પચાસ": 50, "દોઢ": 1.5, "અઢી": 2.5, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}.items()}

def _lookup(table, w):
    w = _nfc(w).lower()
    if w in table:
        return table[w]
    for suf in GU_SUFFIX:  # "લાખની" → "લાખ"
        if w.endswith(suf) and w[: -len(suf)] in table:
            return table[w[: -len(suf)]]
    return None

def _fmt(v):
    return ("%.4f" % v).rstrip("0").rstrip(".")

def norm_numbers(text):
    """Canonical numeric anchors: native digits → ASCII, lakh/crore/हज़ार multipliers, number words
    only when followed by a multiplier (bare एक/એક is too often an article)."""
    toks = [strip_punct(t)[1] for t in text.translate(DIGITS).split()]
    toks = [t for t in toks if t]
    vals, i = set(), 0
    while i < len(toks):
        w, v = toks[i], None
        m = re.match(r"[₹$]?(\d[\d,]*(?:\.\d+)?)", w)
        nxt = _lookup(MULT, toks[i + 1]) if i + 1 < len(toks) else None
        if m:
            try:
                v = float(m.group(1).replace(",", ""))
            except ValueError:
                v = None
        elif _lookup(WORDNUM, w) is not None and nxt is not None:
            v = _lookup(WORDNUM, w)
        if v is not None:
            if nxt is not None:
                v *= nxt
                i += 1
            vals.add(_fmt(v))
        i += 1
    return vals

def gu_to_deva(s):
    # Gujarati block U+0A80–U+0AFF sits 0x180 above Devanagari U+0900–U+097F.
    return "".join(chr(ord(c) - 0x180) if 0x0A80 <= ord(c) <= 0x0AFF else c for c in s)

def trim_gu(w):
    for suf in GU_SUFFIX:
        if w.endswith(suf) and len(w) - len(suf) >= 2:
            return w[: -len(suf)]
    return w

def ent_key(word, lang):
    w = strip_punct(word)[1]
    if not w:
        return ""
    if lang == "gu":
        w = gu_to_deva(trim_gu(w))
    r = romanize(w, "hi") if _INDIC_CHAR.search(w) else w.lower()
    r = re.sub(r"[^a-z0-9]", "", r)
    return (r[0] + re.sub(r"[aeiou]", "", r[1:])) if r else ""

# %% Phase 1 — Data
from datasets import Dataset, concatenate_datasets
from huggingface_hub import hf_hub_download, list_repo_files

def load_xlsum(lang, split):
    name = LANGS[lang]["xlsum"]
    split_name = "validation" if split in ("val", "valid") else split
    repo_files = list_repo_files(
        "csebuetnlp/xlsum",
        repo_type="dataset",
        revision="refs/convert/parquet",
    )
    shards = sorted(
        f for f in repo_files
        if f.startswith(f"{name}/{split_name}/") and f.endswith(".parquet")
    )
    local_paths = [
        hf_hub_download(
            repo_id="csebuetnlp/xlsum",
            repo_type="dataset",
            revision="refs/convert/parquet",
            filename=f,
        )
        for f in shards
    ]
    if len(local_paths) == 1:
        return Dataset.from_parquet(local_paths[0])
    return concatenate_datasets([Dataset.from_parquet(p) for p in local_paths])

P_PILOT, P_FERT, P_SFT = OUT / "pilot.jsonl", OUT / "fert_sentences.jsonl", OUT / "sft_train.jsonl"

if not (P_PILOT.exists() and P_FERT.exists() and P_SFT.exists()):
    rng = random.Random(CFG["seed"])
    pilot, fert, sft = [], [], []
    for lang in LANGS:
        test = load_xlsum(lang, "test")
        rows = []
        for ex in test:
            text, summ = clean(ex["text"], lang), clean(ex["summary"], lang)
            if 150 <= W(text) <= 1500 and 10 <= W(summ) <= 60:
                rows.append(dict(id=ex["id"], text=text, summary=summ))
        # stratify by anchor (number) density tertiles, ~equal draws per tertile
        dens = np.array([len(norm_numbers(r["text"])) / W(r["text"]) for r in rows])
        cuts = np.quantile(dens, [1 / 3, 2 / 3])
        tert = np.digitize(dens, cuts)
        n = CFG["n_pilot_per_lang"]
        chosen = []
        for t, k in zip(range(3), [n - 2 * (n // 3), n // 3, n // 3]):
            idx = [i for i in range(len(rows)) if tert[i] == t]
            chosen += rng.sample(idx, k)
        chosen_set = set(chosen)
        for i in chosen:
            r = rows[i]
            D = truncate_words(r["text"], CFG["max_article_words"])
            pilot.append(dict(doc_id=r["id"], lang=lang, text=D, reference=r["summary"],
                              L_target=int(np.clip(W(r["summary"]), 20, 45)),
                              density_tertile=int(tert[i])))
        # fertility sentences come from test articles that are NOT in the pilot
        sents = []
        for i, r in enumerate(rows):
            if i in chosen_set:
                continue
            sents += [s for s in split_sentences(r["text"]) if 5 <= W(s) <= 60]
        for s in rng.sample(sents, CFG["n_fert_sent"]):
            fert.append(dict(lang=lang, text=s))
        # SFT pairs from train
        train = load_xlsum(lang, "train")
        idx = rng.sample(range(len(train)), min(len(train), CFG["n_sft_per_lang"] * 3))
        k = 0
        for j in idx:
            ex = train[j]
            text, summ = clean(ex["text"], lang), clean(ex["summary"], lang)
            if not (100 <= W(text) and 10 <= W(summ) <= 60):
                continue
            sft.append(dict(lang=lang, text=truncate_words(text, CFG["max_article_words"]),
                            summary=summ, roman=(k % 2 == 1)))
            k += 1
            if k >= CFG["n_sft_per_lang"]:
                break
    jl_write(P_PILOT, pilot); jl_write(P_FERT, fert); jl_write(P_SFT, sft)

pilot, fert_sents, sft_rows = jl_read(P_PILOT), jl_read(P_FERT), jl_read(P_SFT)
print(f"pilot={len(pilot)}  fertility sentences={len(fert_sents)}  sft={len(sft_rows)}")

# %% Phase 2 — Input variants (native / script-mixed / Romanized)
VARIANTS = {"native": lambda t, l: t, "script_mixed": script_mix, "roman": romanize}

for r in pilot:
    r["variants"] = {v: f(r["text"], r["lang"]) for v, f in VARIANTS.items()}
    r["anchors_num"] = sorted(norm_numbers(r["text"]))
for r in fert_sents:
    r["variants"] = {v: f(r["text"], r["lang"]) for v, f in VARIANTS.items()}

ex = pilot[0]
for v in VARIANTS:
    print(f"[{v}] {ex['variants'][v][:220]}\n")
cmi = {l: np.mean([cmi_by_script(r["variants"]["script_mixed"]) for r in fert_sents if r["lang"] == l])
       for l in LANGS}
print("mean CMI of script-mixed fertility sentences:", {k: round(v, 2) for k, v in cmi.items()})

# %% Phase 3 — Tokenizer fertility table
from transformers import AutoTokenizer

def load_tok(hf_id):
    if "IndicBART" in hf_id:
        return AutoTokenizer.from_pretrained(hf_id, do_lower_case=False, use_fast=False, keep_accents=True)
    return AutoTokenizer.from_pretrained(hf_id)

P_FERT_TAB = OUT / "fertility_table.csv"
TOKS = {}
for name, spec in list(CFG["models"].items()) + list(CFG["fert_only"].items()):
    hf_id = spec["hf"] if isinstance(spec, dict) else spec
    try:
        TOKS[name] = load_tok(hf_id)
    except Exception as e:
        print(f"!! tokenizer {hf_id} unavailable ({repr(e)[:150]}) — skipping")

def n_tokens(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"])

_prev = pd.read_csv(P_FERT_TAB) if P_FERT_TAB.exists() else None
_missing = [n for n in TOKS if _prev is None or n not in set(_prev.tokenizer)]
if _missing:  # adds rows for tokenizers unavailable last run (e.g. gated Llama), keeps the rest
    rows = [] if _prev is None else _prev.to_dict("records")
    for name, tok in [(n, TOKS[n]) for n in _missing]:
        for lang in LANGS:
            S = [r for r in fert_sents if r["lang"] == lang]
            for v in VARIANTS:
                T = [n_tokens(tok, r["variants"][v]) for r in S]
                Wd = [max(1, W(r["variants"][v])) for r in S]
                rows.append(dict(tokenizer=name, lang=lang, variant=v,
                                 fertility_corpus=sum(T) / sum(Wd),
                                 fertility_mean=float(np.mean(np.array(T) / np.array(Wd))),
                                 n=len(S)))
    pd.DataFrame(rows).to_csv(P_FERT_TAB, index=False)
fert_tab = pd.read_csv(P_FERT_TAB)
print(fert_tab.pivot_table(index="tokenizer", columns=["lang", "variant"],
                           values="fertility_corpus").round(2).to_string())

# %% Phase 4 — LoRA SFT on Sarvam-1 (resumable every 50 optimizer steps)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, set_peft_model_state_dict

SARVAM = CFG["models"]["sarvam1_sft"]["hf"]
SFT_DIR = OUT / "sarvam1_sft"
SFT_FINAL = SFT_DIR / "final"

def sarvam_prompt(lang, article, n_words):
    return f"Language: {LANGS[lang]['name']}\nLength: {n_words} words\nArticle: {article}\nSummary:"

def fp16_is_safe(model, tok, texts):
    """bf16-trained weights can overflow in fp16 on T4; probe a forward pass before trusting AMP."""
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEV)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        logits = model(**enc).logits
    return bool(torch.isfinite(logits).all())

if not SFT_FINAL.exists():
    tok = AutoTokenizer.from_pretrained(SARVAM)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    examples = []
    for r in sft_rows:
        art = romanize(r["text"], r["lang"]) if r["roman"] else r["text"]
        n = int(np.clip(W(r["summary"]), 20, 45))
        t_ids = tok(" " + r["summary"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        aw = art.split()
        while True:  # shrink the article, never the "Summary:" cue at the end of the prompt
            p_ids = tok(sarvam_prompt(r["lang"], " ".join(aw), n), add_special_tokens=True)["input_ids"]
            if len(p_ids) + len(t_ids) <= CFG["sft_max_len"] or len(aw) < 50:
                break
            aw = aw[: int(len(aw) * 0.9)]
        examples.append((p_ids + t_ids, [-100] * len(p_ids) + t_ids))
    order = list(range(len(examples)))
    random.Random(CFG["seed"]).shuffle(order)

    model = AutoModelForCausalLM.from_pretrained(SARVAM, torch_dtype=torch.float32).to(DEV)
    use_amp = fp16_is_safe(model, tok, [sarvam_prompt("hi", sft_rows[0]["text"][:600], 30)])
    print("fp16 autocast safe for Sarvam-1:", use_amp)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
                                             lora_dropout=CFG["lora_dropout"], task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    mbs, acc = CFG["sft_micro_bs"], CFG["sft_accum"]
    steps_per_epoch = math.ceil(len(examples) / (mbs * acc))
    total = steps_per_epoch * CFG["sft_epochs"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG["sft_lr"],
                            weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * total)), total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, ckpt = 0, SFT_DIR / "ckpt"
    if (ckpt / "state.pt").exists():
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(ckpt / "adapter_model.safetensors")))
        st = torch.load(ckpt / "state.pt", weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        step = st["step"]
        print("resumed SFT at step", step)

    def batch(ids_list):
        L = max(len(x[0]) for x in ids_list)
        inp = torch.full((len(ids_list), L), tok.pad_token_id)
        lab = torch.full((len(ids_list), L), -100)
        att = torch.zeros((len(ids_list), L), dtype=torch.long)
        for i, (a, b) in enumerate(ids_list):
            inp[i, : len(a)] = torch.tensor(a); lab[i, : len(b)] = torch.tensor(b); att[i, : len(a)] = 1
        return inp.to(DEV), att.to(DEV), lab.to(DEV)

    model.train()
    t0, run_loss = time.time(), []
    while step < total:
        base = (step % steps_per_epoch) * mbs * acc
        for a in range(acc):
            sl = order[base + a * mbs: base + (a + 1) * mbs]
            if not sl:
                continue
            inp, att, lab = batch([examples[i] for i in sl])
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(input_ids=inp, attention_mask=att, labels=lab).loss / acc
            scaler.scale(loss).backward()
            run_loss.append(loss.item() * acc)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        step += 1
        if step % 10 == 0:
            print(f"step {step}/{total} loss {np.mean(run_loss[-10 * acc:]):.4f} "
                  f"({(time.time() - t0) / 60:.1f} min)")
        if step % 50 == 0 or step == total:
            model.save_pretrained(ckpt)
            torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(),
                            scaler=scaler.state_dict(), step=step), ckpt / "state.pt")
    model.save_pretrained(SFT_FINAL)
    json.dump(dict(use_amp=use_amp, steps=total, final_loss=float(np.mean(run_loss[-50:]))),
              open(SFT_DIR / "train_info.json", "w"))
    del model, opt
    free_gpu()
print("SFT adapter:", SFT_FINAL)

# %% Phase 5 — Greedy generation: 3 models × {native, roman} × 100 articles
GEN_SCRIPTS = ["native", "roman"]

def chat_prompt(lang, article, n_words, roman):
    L, S = LANGS[lang]["name"], LANGS[lang]["script"]
    note = f"Note: the article is written in Romanized (Latin-script) {L}.\n\n" if roman else ""
    return (f"Summarize the following {L} news article in {L}, written in {S} script, in about "
            f"{n_words} words (1-2 sentences). Output only the summary.\n\n{note}Article:\n{article}")

def load_generator(name):
    spec = CFG["models"][name]
    tok = AutoTokenizer.from_pretrained(spec["hf"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if spec["kind"] == "base_sft":
        model = AutoModelForCausalLM.from_pretrained(spec["hf"], torch_dtype=torch.float32).to(DEV)
        model = PeftModel.from_pretrained(model, str(SFT_FINAL)).merge_and_unload()
        amp = fp16_is_safe(model, tok, [sarvam_prompt("hi", pilot[0]["text"][:600], 30)])
        return model.eval(), tok, amp, "fp32 weights, fp16 autocast" if amp else "fp32"
    # Chat models: try configs in order and keep the first whose logits are finite on a real prompt.
    q4 = lambda cd: BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=cd)
    trials = ([("4bit-nf4 / fp16 storage / fp32 compute", dict(quantization_config=q4(torch.float32), torch_dtype=torch.float16)),
               ("4bit-nf4 / fp32", dict(quantization_config=q4(torch.float32), torch_dtype=torch.float32))]
              if "gemma" in name else
              [("fp16", dict(torch_dtype=torch.float16)),
               ("4bit-nf4 / fp32 compute", dict(quantization_config=q4(torch.float32), torch_dtype=torch.float16))])
    import transformers as tf
    classes = [getattr(tf, c) for c in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                                        "AutoModelForMultimodalLM") if hasattr(tf, c)]
    probe = tok.apply_chat_template([{"role": "user", "content": chat_prompt("gu", pilot[-1]["text"][:800], 30, False)}],
                                    add_generation_prompt=True, return_tensors="pt", return_dict=True)
    for desc, kw in trials:
        for cls in classes:
            try:
                model = cls.from_pretrained(spec["hf"], device_map={"": 0}, **kw).eval()
            except Exception as e:
                print(f"   {cls.__name__} [{desc}] failed: {repr(e)[:160]}")
                continue
            with torch.no_grad():
                lg = model(**{k: v.to(DEV) for k, v in probe.items()}).logits
            if torch.isfinite(lg).all():
                return model, tok, False, desc
            print(f"   non-finite logits with [{desc}], trying next config")
            del model
            free_gpu()
            break
    raise RuntimeError(f"could not load {name} with finite logits")

def clean_output(text, kind):
    text = re.sub(r"(?s)^.*?</think>", "", text)        # drop any reasoning preamble
    if kind == "base_sft":
        text = re.split(r"\n|Language:|Article:", text)[0]
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()

@torch.no_grad()
def generate_one(model, tok, kind, amp, lang, article, n_words, roman):
    if kind == "base_sft":
        enc = tok(sarvam_prompt(lang, article, n_words), return_tensors="pt").to(DEV)
    else:
        enc = tok.apply_chat_template([{"role": "user", "content": chat_prompt(lang, article, n_words, roman)}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True).to(DEV)
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = model.generate(**enc, max_new_tokens=CFG["max_new_tokens"], do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    new = out[0, enc["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True), dt, int(enc["input_ids"].shape[1])

GEN_INFO = OUT / "generator_info.jsonl"
for name, spec in CFG["models"].items():
    path = OUT / f"gen_{name}.jsonl"
    done = {(r["doc_id"], r["script"]) for r in jl_read(path)}
    todo = [(r, s) for r in pilot for s in GEN_SCRIPTS if (r["doc_id"], s) not in done]
    if not todo:
        print(f"{name}: generation complete"); continue
    print(f"== {name}: {len(todo)} summaries to generate")
    try:
        model, tok, amp, desc = load_generator(name)
    except Exception as e:
        print(f"!! skipping {name}: {repr(e)[:300]}")
        continue
    jl_append(GEN_INFO, dict(model=name, hf=spec["hf"], precision=desc))
    torch.cuda.reset_peak_memory_stats()
    for k, (r, s) in enumerate(todo):
        art = r["variants"][s]
        raw, dt, n_in = generate_one(model, tok, spec["kind"], amp, r["lang"], art, r["L_target"], s == "roman")
        summ = clean_output(raw, spec["kind"])
        jl_append(path, dict(model=name, doc_id=r["doc_id"], lang=r["lang"], script=s, summary=summ,
                             raw=raw, latency_s=dt, prompt_tokens=n_in,
                             input_fertility=n_tokens(tok, art) / max(1, W(art)),
                             sum_words=W(summ), sum_tokens=n_tokens(tok, summ),
                             peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9))
        if k % 20 == 0:
            print(f"  [{k}/{len(todo)}] {r['lang']}/{s} {dt:.1f}s :: {summ[:120]}")
    del model
    free_gpu()

# %% Phase 6a — Translation of sources and claims (IndicTrans2, NLLB fallback)
P_MT = OUT / "mt_cache.jsonl"
MT_CACHE = {r["k"]: r["en"] for r in jl_read(P_MT)}
_hash = lambda lang, s: hashlib.md5(f"{lang}|{s}".encode()).hexdigest()

GENS = [g for name in CFG["models"] for g in jl_read(OUT / f"gen_{name}.jsonl")]
PILOT = {(r["lang"], r["doc_id"]): r for r in pilot}

def claims_of(summary):
    return [c for c in split_sentences(summary) if W(c) >= 3]

need = {}
for r in pilot:
    for s in split_sentences(r["text"]):
        need[_hash(r["lang"], s)] = (r["lang"], s)
for g in GENS:
    for c in claims_of(g["summary"]):
        need[_hash(g["lang"], c)] = (g["lang"], c)
todo = [(k, l, s) for k, (l, s) in need.items() if k not in MT_CACHE]
print(f"MT: {len(todo)} sentences to translate ({len(MT_CACHE)} cached)")

if todo:
    from transformers import AutoModelForSeq2SeqLM
    mt_kind = "indictrans2"
    try:
        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:
            from IndicTransToolkit import IndicProcessor
        ip = IndicProcessor(inference=True)
        mt_tok = AutoTokenizer.from_pretrained(CFG["mt"], trust_remote_code=True)
        mt = AutoModelForSeq2SeqLM.from_pretrained(CFG["mt"], trust_remote_code=True).to(DEV).eval()
    except Exception as e:
        print("IndicTrans2 unavailable, using NLLB-200:", repr(e)[:200])
        mt_kind = "nllb"
        mt_tok = AutoTokenizer.from_pretrained(CFG["mt_fallback"])
        mt = AutoModelForSeq2SeqLM.from_pretrained(CFG["mt_fallback"]).to(DEV).eval()

    @torch.no_grad()
    def translate(sents, lang):
        src = LANGS[lang]["it2"]
        if mt_kind == "indictrans2":
            b = ip.preprocess_batch(sents, src_lang=src, tgt_lang="eng_Latn")
            enc = mt_tok(b, truncation=True, padding="longest", return_tensors="pt").to(DEV)
            try:
                out = mt.generate(**enc, num_beams=4, max_length=256, use_cache=True)
            except Exception:
                out = mt.generate(**enc, num_beams=4, max_length=256, use_cache=False)
            dec = mt_tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return ip.postprocess_batch(dec, lang="eng_Latn")
        mt_tok.src_lang = src
        enc = mt_tok(sents, truncation=True, padding="longest", return_tensors="pt").to(DEV)
        out = mt.generate(**enc, num_beams=4, max_length=256,
                          forced_bos_token_id=mt_tok.convert_tokens_to_ids("eng_Latn"))
        return mt_tok.batch_decode(out, skip_special_tokens=True)

    for lang in LANGS:
        items = [(k, s) for k, l, s in todo if l == lang]
        for i in range(0, len(items), 16):
            chunk = items[i: i + 16]
            en = translate([s for _, s in chunk], lang)
            for (k, _), e in zip(chunk, en):
                MT_CACHE[k] = e
                jl_append(P_MT, dict(k=k, en=e, mt=mt_kind))
            if i % 320 == 0:
                print(f"  {lang}: {i}/{len(items)}")
    del mt
    free_gpu()

def en_of(lang, s):
    return MT_CACHE.get(_hash(lang, s), "")

# %% Phase 6b — Claim verification: MiniCheck-FT5 (English), mDeBERTa-XNLI (native), round-trip router
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

def chunks(text, size, stride):
    w = text.split()
    if len(w) <= size:
        return [" ".join(w)]
    return [" ".join(w[i: i + size]) for i in range(0, len(w) - stride, stride)]

P_CLAIMS = OUT / "claim_scores.jsonl"
scored = {(r["model"], r["doc_id"], r["script"], r["ci"]) for r in jl_read(P_CLAIMS)}
claim_jobs = []
for g in GENS:
    for ci, c in enumerate(claims_of(g["summary"])):
        if (g["model"], g["doc_id"], g["script"], ci) not in scored:
            claim_jobs.append((g, ci, c))
print("claims to score:", len(claim_jobs))

if claim_jobs:
    mc_tok = AutoTokenizer.from_pretrained(CFG["minicheck"])
    mc = AutoModelForSeq2SeqLM.from_pretrained(CFG["minicheck"], torch_dtype=torch.float32).to(DEV).eval()
    # MiniCheck-FT5 scores the first decoder step over tokens 3 ('▁', which starts "0") and 209 ('1').
    MC_IDS = [3, 209]
    print("MiniCheck label tokens:", mc_tok.convert_ids_to_tokens(MC_IDS))
    nli_tok = AutoTokenizer.from_pretrained(CFG["nli"])
    nli = AutoModelForSequenceClassification.from_pretrained(CFG["nli"]).to(DEV).eval()
    ENT = [i for i, l in nli.config.id2label.items() if l.lower().startswith("entail")][0]

    @torch.no_grad()
    def minicheck(doc_chunks, claim):
        texts = [f"premise: {d} hypothesis: {claim}" for d in doc_chunks]
        best = 0.0
        for i in range(0, len(texts), 8):
            enc = mc_tok(texts[i: i + 8], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEV)
            dec = torch.full((enc["input_ids"].shape[0], 1), mc.config.decoder_start_token_id, device=DEV)
            lg = mc(**enc, decoder_input_ids=dec).logits[:, 0, MC_IDS]
            best = max(best, float(torch.softmax(lg, -1)[:, 1].max()))
        return best

    @torch.no_grad()
    def nli_entail(doc_chunks, claim):
        enc = nli_tok(doc_chunks, [claim] * len(doc_chunks), return_tensors="pt", padding=True,
                      truncation="only_first", max_length=512).to(DEV)
        return float(torch.softmax(nli(**enc).logits, -1)[:, ENT].max())

    _p = "The district hospital in Surat admitted 42 patients on Monday after a gas leak."
    sane_hi, sane_lo = minicheck([_p], "42 patients were admitted to a hospital in Surat."), \
        minicheck([_p], "The gas leak killed 300 people in Delhi.")
    print(f"MiniCheck sanity: supported={sane_hi:.2f} (want >0.5)  unsupported={sane_lo:.2f} (want <0.5)")
    assert sane_hi > 0.5 > sane_lo, "MiniCheck label-token mapping looks wrong; check MC_IDS"

    src_cache = {}
    for n, (g, ci, c) in enumerate(claim_jobs):
        key = (g["lang"], g["doc_id"])
        if key not in src_cache:
            D = PILOT[key]["text"]
            en_doc = " ".join(en_of(g["lang"], s) for s in split_sentences(D))
            src_cache[key] = (chunks(en_doc, 250, 125), chunks(D, 200, 100), norm_numbers(D))
        en_chunks, nat_chunks, _ = src_cache[key]
        c_en = en_of(g["lang"], c)
        p_mc = minicheck(en_chunks, c_en) if c_en else 0.0
        p_nli = nli_entail(nat_chunks, c)
        a_nat, a_en = norm_numbers(c), norm_numbers(c_en)
        survived = a_nat <= a_en           # round-trip anchor check (numbers)
        p_router = p_mc if survived else p_nli
        jl_append(P_CLAIMS, dict(model=g["model"], doc_id=g["doc_id"], lang=g["lang"], script=g["script"],
                                 ci=ci, claim=c, claim_en=c_en, p_minicheck=p_mc, p_nli=p_nli,
                                 route="minicheck" if survived else "nli", p_router=p_router))
        if n % 100 == 0:
            print(f"  {n}/{len(claim_jobs)} mc={p_mc:.2f} nli={p_nli:.2f} :: {c_en[:90]}")
    del mc, nli
    free_gpu()

# %% Phase 6c — Anchors (IndicNER entities + numbers) and Indic-tokenized ROUGE
from transformers import pipeline
from rapidfuzz.distance import Levenshtein
from rouge_score import rouge_scorer
from indicnlp.tokenize import indic_tokenize

P_ANCH = OUT / "summary_anchor_rouge_v2.jsonl"   # v1 had matra-stripped entity strings
have = {(r["model"], r["doc_id"], r["script"]) for r in jl_read(P_ANCH)}
jobs = [g for g in GENS if (g["model"], g["doc_id"], g["script"]) not in have]

class IndicTok:
    def tokenize(self, text):
        return [t for t in indic_tokenize.trivial_tokenize(text) if not all(ch in PUNCT for ch in t)]

if jobs:
    ner = pipeline("token-classification", model=CFG["ner"], aggregation_strategy="first", device=0)

    def _span(S, a, b):  # widen to whole whitespace-delimited words
        while a > 0 and not S[a - 1].isspace():
            a -= 1
        while b < len(S) and not S[b].isspace():
            b += 1
        return S[a:b]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=IndicTok())
    src_keys = {}
    for g in jobs:
        key = (g["lang"], g["doc_id"])
        D = PILOT[key]["text"]
        if key not in src_keys:
            src_keys[key] = ({ent_key(w, g["lang"]) for w in D.split()} - {""}, norm_numbers(D))
        keys, nums = src_keys[key]
        S = g["summary"]
        s_nums = norm_numbers(S)
        # e["word"] comes from the NER tokenizer, which drops vowel signs (वेस्टइंडीज → सटइडीज); use offsets
        ents = [_span(S, e["start"], e["end"]) for e in (ner(S) if S else [])
                if e.get("entity_group") in ("PER", "LOC", "ORG")]
        ents = list(dict.fromkeys(e.replace("##", "").strip() for e in ents if e.strip()))

        def ent_supported(e):
            for w in e.split():
                k = ent_key(w, g["lang"])
                if not k:
                    continue
                if k in keys:
                    continue
                if len(k) <= 2 or not any(Levenshtein.distance(k, sk, score_cutoff=1) <= 1 for sk in keys):
                    return False
            return True

        sup_e = [ent_supported(e) for e in ents]
        sup_n = [n in nums for n in s_nums]
        n_anch = len(sup_e) + len(sup_n)
        rs = scorer.score(PILOT[key]["reference"], S) if S else None
        jl_append(P_ANCH, dict(model=g["model"], doc_id=g["doc_id"], script=g["script"],
                               n_anchors=n_anch, n_unsup_anchors=n_anch - sum(sup_e) - sum(sup_n),
                               entities=ents, ent_supported=sup_e, numbers=sorted(s_nums), num_supported=sup_n,
                               rouge1=rs["rouge1"].fmeasure * 100 if rs else 0.0,
                               rouge2=rs["rouge2"].fmeasure * 100 if rs else 0.0,
                               rougeL=rs["rougeL"].fmeasure * 100 if rs else 0.0))
    del ner
    free_gpu()

# %% Phase 7 — Assemble per-summary table
claims = pd.DataFrame(jl_read(P_CLAIMS))
tau = CFG["tau_ver"]
agg = (claims.assign(s_router=claims.p_router >= tau, s_mc=claims.p_minicheck >= tau, s_nli=claims.p_nli >= tau)
       .groupby(["model", "doc_id", "script"])
       .agg(n_claims=("ci", "size"), P_router=("s_router", "mean"), P_mc=("s_mc", "mean"),
            P_nli=("s_nli", "mean"), frac_routed_nli=("route", lambda x: (x == "nli").mean()))
       .reset_index())
gens = pd.DataFrame(GENS)
anch = pd.DataFrame(jl_read(P_ANCH))
df = gens.merge(agg, on=["model", "doc_id", "script"], how="left").merge(anch, on=["model", "doc_id", "script"], how="left")
df["src_words"] = [W(PILOT[(l, d)]["text"]) for l, d in zip(df.lang, df.doc_id)]
df["script_ok"] = [script_fraction(s, l) >= 0.5 for s, l in zip(df.summary, df.lang)]
df["has_claims"] = df.n_claims.fillna(0) > 0
df["halluc"] = 1 - df.P_router
df["halluc_mc"] = 1 - df.P_mc
df["halluc_nli"] = 1 - df.P_nli
df["anchor_err"] = np.where(df.n_anchors > 0, df.n_unsup_anchors / df.n_anchors.where(df.n_anchors > 0), np.nan)
df["CR"] = 100 * df.sum_words / df.src_words
df["doc_key"] = df.lang + "_" + df.doc_id
df["cell"] = df.doc_key + "_" + df.script
df["log_len"] = np.log(df.sum_words.clip(lower=1))
df.to_csv(OUT / "h1_per_summary.csv", index=False)

summary = (df.groupby(["model", "lang", "script"])
           .agg(n=("doc_id", "size"), script_adherence=("script_ok", "mean"),
                fertility=("input_fertility", "mean"), halluc=("halluc", "mean"),
                halluc_minicheck=("halluc_mc", "mean"), halluc_nli=("halluc_nli", "mean"),
                anchor_err=("anchor_err", "mean"), rouge1=("rouge1", "mean"), rouge2=("rouge2", "mean"),
                rougeL=("rougeL", "mean"), words=("sum_words", "mean"), tokens=("sum_tokens", "mean"),
                CR_pct=("CR", "mean"), latency_s=("latency_s", "mean"), peak_vram_gb=("peak_vram_gb", "max"))
           .round(3))
print(summary.to_string())

# %% Phase 7b — Statistics for H1 and the gate decision
import statsmodels.formula.api as smf
from scipy import stats

A = df[df.script_ok & df.has_claims].copy()   # main analysis: gradeable, correct-script outputs

def fe_ols(formula, data, outcome="halluc"):
    try:
        return _fe_ols(formula, data, outcome)
    except Exception as e:  # e.g. only one model loaded → across-model spec is not identified
        return dict(error=repr(e)[:200])

def _fe_ols(formula, data, outcome):
    d = data.dropna(subset=[outcome, "input_fertility", "log_len"])
    m = smf.ols(formula.replace("Y", outcome), data=d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.doc_key)[0]})
    b, p2 = m.params["input_fertility"], m.pvalues["input_fertility"]
    return dict(beta=float(b), se=float(m.bse["input_fertility"]), p_two=float(p2),
                p_one=float(p2 / 2 if b > 0 else 1 - p2 / 2), n=int(m.nobs))

tests = {}
for outcome in ["halluc", "halluc_mc", "halluc_nli", "anchor_err"]:
    tests[outcome] = {
        # all variation, holding the article fixed
        "pooled_docFE": fe_ols("Y ~ input_fertility + log_len + C(doc_key)", A, outcome),
        # within-model: same model, same article, fertility varies only through script
        "within_model": fe_ols("Y ~ input_fertility + log_len + C(doc_key) + C(model)", A, outcome),
        # across-family: same article and same input script, fertility varies only through model
        "across_model": fe_ols("Y ~ input_fertility + log_len + C(cell)", A, outcome),
    }
rho = stats.spearmanr(A.input_fertility, A.halluc)
tests["spearman_pooled"] = dict(rho=float(rho.statistic), p=float(rho.pvalue), n=len(A))

# paired native→roman shifts per model (Wilcoxon)
paired = {}
for m in A.model.unique():
    piv = A[A.model == m].pivot_table(index="doc_key", columns="script", values=["halluc", "input_fertility"])
    piv = piv.dropna()
    if len(piv) >= 10 and ("halluc", "roman") in piv.columns and ("halluc", "native") in piv.columns:
        dh = piv["halluc"]["roman"] - piv["halluc"]["native"]
        dF = piv["input_fertility"]["roman"] - piv["input_fertility"]["native"]
        w = stats.wilcoxon(dh) if (dh != 0).any() else None
        paired[m] = dict(n=len(piv), mean_dF=float(dF.mean()), mean_dHalluc=float(dh.mean()),
                         wilcoxon_p=float(w.pvalue) if w else 1.0)
tests["paired_native_vs_roman"] = paired

def paired_bootstrap(a_model, b_model, col="halluc", B=1000, seed=42):
    x = A[A.model == a_model].set_index("cell")[col]
    y = A[A.model == b_model].set_index("cell")[col]
    common = x.index.intersection(y.index)
    d = (x[common] - y[common]).values
    if len(d) < 10:
        return None
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, len(d)).mean() for _ in range(B)])
    return dict(n=len(d), mean_diff=float(d.mean()), ci95=[float(np.percentile(boots, 2.5)),
                                                            float(np.percentile(boots, 97.5))])

contrasts = {f"{a}-{b}": paired_bootstrap(a, b) for a, b in
             [("sarvam1_sft", "gemma4_e2b"), ("sarvam1_sft", "llama32_3b"), ("gemma4_e2b", "llama32_3b")]}
tests["model_contrasts_halluc"] = contrasts

# ---- Gate decision
def mstat(m, col):
    s = df[df.model == m][col]
    return float(s.mean()) if len(s) else float("nan")

def _sig(t):
    return bool("beta" in t and t["beta"] > 0 and t["p_one"] < 0.05)

# H1 claims the association holds within a model across scripts AND across model families. The pooled
# spec alone is not enough: Romanization raises both fertility and hallucination, so it confounds them.
h1_within, h1_across = _sig(tests["halluc"]["within_model"]), _sig(tests["halluc"]["across_model"])
h1_supported = h1_within and h1_across
sg = contrasts.get("sarvam1_sft-gemma4_e2b")
sarvam_competitive = bool(sg and sg["ci95"][1] <= 0.05)  # not more than 5 points worse than Gemma-4
sarvam_quality_ok = bool(mstat("sarvam1_sft", "script_ok") >= 0.90 and
                         mstat("sarvam1_sft", "rougeL") >= 0.8 * mstat("gemma4_e2b", "rougeL"))
keep_sarvam = h1_supported and sarvam_competitive and sarvam_quality_ok
decision = dict(
    h1_supported=h1_supported, h1_within_model=h1_within, h1_across_model=h1_across,
    h1_pooled_only=_sig(tests["halluc"]["pooled_docFE"]),
    sarvam_competitive=sarvam_competitive, sarvam_quality_ok=sarvam_quality_ok,
    mt_engine={str(k): int(v) for k, v in pd.Series([r.get("mt") for r in jl_read(P_MT)]).value_counts().items()},
    backbone="sarvamai/sarvam-1 (+LoRA SFT)" if keep_sarvam else CFG["models"]["gemma4_e2b"]["hf"],
    rule=("keep Sarvam-1 iff H1 [β_fertility>0, one-sided p<0.05, in BOTH the within-model "
          "(doc+model FE) and across-model (doc×script FE) specs] AND "
          "(Sarvam−Gemma hallucination 95% CI upper bound ≤ +0.05) AND "
          "(Sarvam script adherence ≥ 0.90 AND ROUGE-L ≥ 0.8×Gemma)"),
)
tests["decision"] = decision
json.dump(tests, open(OUT / "h1_tests.json", "w"), indent=2, default=str)
print(json.dumps(decision, indent=2))

# %% Phase 7c — Plot + markdown report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for (m, s), g in A.groupby(["model", "script"]):
    ax[0].scatter(g.input_fertility, g.halluc, s=12, alpha=0.5, label=f"{m}/{s}")
ax[0].set_xlabel("input fertility (tokens / word)"); ax[0].set_ylabel("claim hallucination rate (router)")
ax[0].legend(fontsize=7)
cell = A.groupby(["model", "lang", "script"]).agg(F=("input_fertility", "mean"), H=("halluc", "mean")).reset_index()
for _, r in cell.iterrows():
    ax[1].scatter(r.F, r.H, s=60)
    ax[1].annotate(f"{r.model}\n{r.lang}/{r.script}", (r.F, r.H), fontsize=7)
ax[1].set_xlabel("mean input fertility"); ax[1].set_ylabel("mean hallucination rate")
fig.tight_layout(); fig.savefig(OUT / "h1_fertility_vs_halluc.png", dpi=150)

gen_info = pd.DataFrame(jl_read(GEN_INFO)).drop_duplicates("model", keep="last")
with open(OUT / "H1_REPORT.md", "w", encoding="utf-8") as f:
    f.write("# FactIndic — Experiment 0 (H1 Gate) Report\n\n")
    f.write(f"Pilot: {len(pilot)} XL-Sum test articles ({CFG['n_pilot_per_lang']} hi + "
            f"{CFG['n_pilot_per_lang']} gu), truncated to {CFG['max_article_words']} words, greedy decoding.\n\n")
    f.write("## Gate decision\n\n```json\n" + json.dumps(decision, indent=2) + "\n```\n\n")
    f.write("## Tokenizer fertility (tokens/word, corpus level, 1,000 sentences per language)\n\n")
    f.write(fert_tab.pivot_table(index="tokenizer", columns=["lang", "variant"],
                                 values="fertility_corpus").round(2).to_markdown() + "\n\n")
    f.write("## Generators\n\n" + gen_info.to_markdown(index=False) + "\n\n")
    f.write("## Per model × language × input script\n\n" + summary.to_markdown() + "\n\n")
    f.write("## H1 regressions (cluster-robust SE by article; β on input fertility)\n\n")
    reg = pd.DataFrame([dict(outcome=o, spec=s, **v) for o in ["halluc", "halluc_mc", "halluc_nli", "anchor_err"]
                        for s, v in tests[o].items()])
    f.write(reg.round(4).to_markdown(index=False) + "\n\n")
    f.write(f"Spearman ρ (pooled): {tests['spearman_pooled']}\n\n")
    f.write("## Paired native→Romanized shifts\n\n```json\n" + json.dumps(paired, indent=2) + "\n```\n\n")
    f.write("## Model contrasts (paired bootstrap, 1,000 resamples)\n\n```json\n" +
            json.dumps(contrasts, indent=2) + "\n```\n\n")
    f.write("![fertility vs hallucination](h1_fertility_vs_halluc.png)\n")
print("Wrote", OUT / "H1_REPORT.md")
print("Download the whole folder:", OUT)

# %%
# !zip -r /kaggle/working/factindic_h1.zip /kaggle/working/factindic_h1
