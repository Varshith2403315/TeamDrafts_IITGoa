"""Render the slide sources in docs/deck/slides/ (one <section> per slide) to docs/DoseTwin_presentation.pdf."""
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "deck"

order = json.loads((DECK / "deck.json").read_text())["order"]
slides = []
for sid in order:
    h = (DECK / "slides" / f"{sid}.html").read_text()
    h = re.sub(r"<aside>.*?</aside>", "", h, flags=re.S)
    h = h.replace("../../figures/", (ROOT / "docs" / "figures").as_uri() + "/")
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
