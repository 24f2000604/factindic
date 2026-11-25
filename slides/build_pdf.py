"""Renders the deck to slides/FactIndic_slides.pdf with headless Chrome: python slides/build_pdf.py
The deck itself is a Slides artifact on claude.ai; these are its source files (speaker notes are in <aside>)."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = ("*{box-sizing:border-box}html,body{margin:0}h1,h2,h3,p,ul{margin:0}section{position:relative;width:1920px;height:1080px;"
        "overflow:hidden;break-after:page}section:last-child{break-after:auto}aside{display:none}table{border-collapse:collapse}"
        "td,th{padding:.35em .6em;border-bottom:1px solid #c9d1dc;text-align:left}*{-webkit-print-color-adjust:exact}"
        "@page{size:1920px 1080px;margin:0}")
d = json.loads((HERE / "deck.json").read_text(encoding="utf-8"))
page = "<!doctype html><html lang='en'><meta charset='utf-8'><title>" + d["title"] + "</title>"
page += "".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in d["faces"].values())
page += f"<style>{BASE}{(HERE / 'deck.css').read_text(encoding='utf-8')}</style>"
page += "".join((HERE / "slides" / f"{i}.html").read_text(encoding="utf-8") for i in d["order"])
chrome = next(filter(None, map(shutil.which, ("google-chrome-stable", "google-chrome", "chromium"))), None)
if not chrome:
    sys.exit("Chrome or Chromium is needed")
with tempfile.TemporaryDirectory() as t:
    f = Path(t) / "deck.html"
    f.write_text(page, encoding="utf-8")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", f"--user-data-dir={t}/u", "--no-pdf-header-footer", "--virtual-time-budget=9000",
                    f"--print-to-pdf={HERE / 'FactIndic_slides.pdf'}", f.as_uri()], check=True, capture_output=True, timeout=90)
print("ok")
