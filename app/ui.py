"""Streamlit front end.  streamlit run app/ui.py   (FACTINDIC_API points at the FastAPI service)"""
import html
import os

import requests
import streamlit as st

API = os.environ.get("FACTINDIC_API", "http://127.0.0.1:8000")
st.set_page_config(page_title="FactIndic", page_icon="📝", layout="centered")
st.title("FactIndic")
st.caption("Factuality-aware summaries of Hindi, Gujarati, Hinglish, Gujlish and English documents. Everything runs on this computer.")

try:
    h = requests.get(f"{API}/health", timeout=5).json()
except requests.RequestException:
    st.error(f"The API at {API} does not answer. Start it with ./run_app.sh")
    st.stop()
if not h["ready"]:
    st.error(h["error"]) if h["error"] else st.info(f"The models are loading ({h['stage']}). This takes about a minute.")
    st.button("Check again")
    st.stop()

with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Document language", ["Auto", "Hindi", "Gujarati", "Hinglish", "Gujlish", "English"])
    mode = st.radio("Model", ["full", "fast"], format_func={"full": "FactIndic (full pipeline)", "fast": "Fast (anchor prompt only)"}.get,
                    help="Full: five candidate summaries, checked against the document and reranked. Fast: one unchecked summary.")
    words = st.slider("Summary length (words)", 25, 100, 40, 5)
    st.caption("Hinglish is summarized in Hindi (Devanagari), Gujlish in Gujarati script.")

up, paste = st.tabs(["Upload a file", "Paste text"])
file = up.file_uploader("PDF, Word or text file", type=["pdf", "docx", "txt", "md"])
text = paste.text_area("Document", height=240, placeholder="यहाँ लेख चिपकाएँ · લેખ અહીં પેસ્ટ કરો · lekh yahan paste karein")
if file and text.strip():
    st.caption("Both a file and text are given: the file is used.")

if st.button("Summarize", type="primary", disabled=not (file or text.strip())):
    opts = dict(language=language, mode=mode, words=words)
    with st.spinner("Reading, writing and checking the summary…" if mode == "full" else "Writing the summary…"):
        try:
            if file:
                r = requests.post(f"{API}/summarize/file", data=opts, files={"file": (file.name, file.getvalue())}, timeout=600)
            else:
                r = requests.post(f"{API}/summarize", json=dict(opts, text=text), timeout=600)
        except requests.RequestException as e:
            st.error(f"The API did not answer ({type(e).__name__}).")
            st.stop()
    if r.ok:
        st.session_state["res"] = r.json()
    else:
        st.session_state.pop("res", None)
        d = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else r.text[:300]
        st.error(d if isinstance(d, str) else "; ".join(x.get("msg", "") for x in d))

res = st.session_state.get("res")
if res:
    st.subheader("Summary")
    st.markdown(f"<div style='font-size:1.2rem;line-height:1.8;padding:1rem;border:1px solid rgba(128,128,128,.35);"
                f"border-radius:.5rem'>{html.escape(res['summary'])}</div>", unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Document", res["language"])
    b.metric("Input script", {"native": "native", "roman": "Romanized", "script_mixed": "mixed"}[res["variant"]])
    c.metric("Words", f"{res['document_words']} → {res['words']}")
    d.metric("Time", f"{res['seconds']['total']:.1f} s")
    for n in res["notes"]:
        st.warning(n)
    st.download_button("Download the summary", res["summary"], "summary.txt")
    if res["sentences"]:
        st.subheader("Fact check")
        for s in res["sentences"]:
            mark = "🗑️ removed" if s["removed"] else ("✅ supported" if s["supported"] else "⚠️ not verified")
            st.markdown(f"**{mark}** · score {s['score']:.2f} · {s['checked_by']}")
            st.text(s["text"])
            if s["english"] != s["text"]:
                st.caption("English: " + s["english"])
        st.caption("The check is automatic and can be wrong in both directions.")
    with st.expander("Names, numbers and dates found in the document"):
        for k in ("entities", "numbers", "dates"):
            st.text(f"{k.capitalize()}: " + (", ".join(res["anchors"][k]) or "none"))
    if res["candidates"]:
        with st.expander("All candidate summaries"):
            for x in res["candidates"]:
                st.markdown(f"**Candidate {x['k']}** ({x['source']}){' — chosen' if x['k'] == res['chosen'] else ''} · J = {x['J']:.2f} · "
                            f"supported {x['supported']:.0%} · anchor recall {x['anchor_recall']:.0%}")
                st.text(x["summary"])
elif not file and not text.strip():
    st.info("Upload a document or paste text, then press Summarize.")
