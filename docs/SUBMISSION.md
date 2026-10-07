# Submission checklist (Phase 1 deadline: 20 October 2026)

The Unstop form asks only for the team leader's name, phone number, email and the public GitHub link, so the repository is the submission.

## Links
- Repository: https://github.com/Varshith2403315/TeamDrafts_IITGoa
- Live dashboard (GitHub Pages, served from `docs/`): https://varshith2403315.github.io/TeamDrafts_IITGoa/
- Demo video (unlisted YouTube, at least 20 min): add to the README table once uploaded

## Required items (from the challenge rules)
- [x] Team details and college/incubator information (README)
- [x] Project title
- [x] Problem statement and healthcare use case
- [x] Technical stack, AI/ML model and framework details
- [ ] Demo video, at least 20 minutes (script: `docs/VIDEO_SCRIPT.md`)
- [x] Open-source licence (MIT)
- [x] Architecture diagram in PDF (`docs/DoseTwin_architecture.pdf`)
- [x] Presentation in PDF (`docs/DoseTwin_presentation.pdf`)
- [x] Repository public and named "Team Name_College Name"

## Before submitting
- [ ] Video link added to the README and plays when logged out
- [ ] Live dashboard and README images open in a private browser window
- [ ] Unstop form: leader name, phone, email, GitHub link

## Rebuilding the dashboard or slides
```bash
python scripts/build_web.py        # web/dist/ and docs/index.html (GitHub Pages)
python scripts/render_deck_pdf.py  # docs/deck/slides/*.html -> docs/DoseTwin_presentation.pdf
```
