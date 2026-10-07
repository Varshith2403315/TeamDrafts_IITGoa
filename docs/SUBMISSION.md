# Submission kit (Phase 1 deadline: 20 October 2026)

The platform asks only for **the team leader's name, phone number, email and the public GitHub link**. Everything else must be inside the repository, so the README is the submission.

## 1. Fill in the placeholders (10 min)
- `README.md` table: team name, members (leader first), college/incubator, video link, live-demo link.
- Deck cover and closing slides: `[Team name]`, `[Member names]`, links. Edit them in the deck page, export it as PDF, and save it as `docs/DoseTwin_presentation.pdf`, replacing the auto-rendered copy. Or edit `docs/deck/slides/*.html` and run `python scripts/render_deck_pdf.py`.

## 2. Push the repository with the required name (5 min)
The rules ask for the name format "Team Name_College Name", e.g. `TeamName_IITGoa`.
```bash
# unzip DoseTwin_repo.zip, rename the folder to TeamName_IITGoa, then inside it:
git add -A && git commit -m "docs: add team details and links"
# create an EMPTY public repo on github.com called TeamName_IITGoa, then:
git remote add origin https://github.com/Varshith2403315/TeamDrafts_IITGoa.git
git branch -M main && git push -u origin main
```

## 3. Make every link public
- Live dashboard and deck pages: use **Share** → anyone with the link can view.
- YouTube video: **Unlisted** (not Private).
- Check each link in a private browser window.

## 4. Record the video (≥ 20 min)
Follow `docs/VIDEO_SCRIPT.md` (about 22.5 min).

## 5. Final checklist
- [ ] README has team details, college, title, problem and use case, stack, model details, video link, licence
- [ ] `docs/DoseTwin_architecture.pdf` and `docs/DoseTwin_presentation.pdf` are in the repo
- [ ] Repo is public; README images render on GitHub
- [ ] Video is at least 20:00 and plays when logged out
- [ ] Submitted on Unstop: leader name, phone, email, GitHub link

## Paste-ready text (for the README or your own reference)
**Problem:** insulin dosing on pens is manual and error-prone; affordable CGMs in India alarm only after glucose is already low; nocturnal hypoglycaemia is the most feared adverse event.
**Solution:** a personal glucose–insulin twin fusing a synthetic EHR (demographics, diagnoses, labs incl. HbA1c, genetics, prescriptions) with CGM, smart-pen, meal-log and step data. It predicts hypoglycaemia and tests every dose 5 h ahead. A clinician dashboard lets the doctor try new settings on the virtual patient.
**Result (in silico, held-out, 90 patient-weeks):** −69% time below 70 mg/dL, −72% nocturnal lows and +4.6 points time in range vs a threshold-alarm CGM with a bolus calculator (p < 0.001).
