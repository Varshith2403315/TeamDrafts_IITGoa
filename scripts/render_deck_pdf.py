"""Render the slide deck (exported slide HTML in docs/deck/) to docs/DoseTwin_presentation.pdf."""
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "deck"
BLOBS = {"1c3aeea7da0789f0954c8d9d1fd99057": "arms.png", "bc3248e3fa48e9946912be07287a46a1": "alerts.png",
         "22436f6944cd6ff8ed95b38a6410bd8c": "forecast.png", "5539ca7fabf6867935eda1492fbaa888": "clinic_view.png",
         "f560c77db5211a783d9e0aab19006063": "demo_sim.png"}
ARROW = ('<svg width="72" height="36" viewBox="0 0 72 36" style="flex:none"><path d="M0,10 H43 V0 L72,18 L43,36 V26 H0 Z" '
         'fill="#8794a1"/></svg>')

order = json.loads((DECK / "deck.json").read_text())["order"]
slides = []
for sid in order:
    h = (DECK / "slides" / f"{sid}.html").read_text()
    h = re.sub(r"<aside>.*?</aside>", "", h, flags=re.S)
    h = re.sub(r'<x-shape kind="arrow-right"[^>]*></x-shape>', ARROW, h)
    for k, v in BLOBS.items():
        h = h.replace(f"/_blob/{k}", (ROOT / "docs" / "figures" / v).as_uri())
    h = h.replace("<section ", '<section class="s" ', 1)
    slides.append(h)
html = ('<!doctype html><html><head><meta charset="utf-8">'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
        '<style>@page{size:1920px 1080px;margin:0}*{box-sizing:border-box}body{margin:0}'
        '.s{width:1920px;height:1080px;position:relative;overflow:hidden;page-break-after:always}'
        'h1,h2,h3,p,ul,table{margin:0}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #dbe2e8;padding:.35em .6em;text-align:left}'
        'th{font-weight:600}</style></head><body>' + "".join(slides) + "</body></html>")
tmp = DECK / "_render.html"
tmp.write_text(html)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto(tmp.as_uri()); pg.wait_for_timeout(1500)
    pg.pdf(path=str(ROOT / "docs" / "DoseTwin_presentation.pdf"), width="1920px", height="1080px", print_background=True)
    pg.screenshot(path=str(DECK / "_preview.png"), full_page=True)
    b.close()
tmp.unlink()
print("wrote docs/DoseTwin_presentation.pdf with", len(order), "slides")
