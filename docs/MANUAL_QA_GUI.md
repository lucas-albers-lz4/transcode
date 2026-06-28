# Manual QA checklist — GUI wizard

Run on **dev build** (`python transcode_gui.py`) and again on **frozen build** (`dist/transcode_gui/transcode_gui`).
Use test folders only if a production transcode job is running elsewhere.

## Prerequisites

- [ ] `./scripts/check_prerequisites.sh` reports ffmpeg and ffprobe OK
- [ ] `pytest tests/ -q` passes

## Wizard

| Area | Steps | Pass |
|------|-------|------|
| FFmpeg gate | Temporarily hide ffmpeg from PATH; launch app → install hint + Quit | |
| Step 1 | Browse source + destination; Next triggers scan with progress | |
| Step 2 | Switch Archive / Fast / Quality; space slider updates lines | |
| Step 2 | “Enough space” shows time-to-complete line when space OK | |
| Convert | Start; status shows `Converting… X / Y completed` | |
| Details | Show details → log appears; Hide details stays above log | |
| Cancel/resume | Convert 2 files, Cancel, Start again → `2 / Y` then continues | |
| Done | Success message; Open output folder works | |
| Theme | Readable text (light/dark); rectangular buttons; Georgia headers | |

## Frozen build only

| Area | Steps | Pass |
|------|-------|------|
| Launch | `./dist/transcode_gui/transcode_gui` opens wizard | |
| Convert | One short test clip converts end-to-end | |

## Accessibility (v1 — OS tools)

- [ ] OS magnifier/zoom usable while app is open
- [ ] System light/dark mode both readable
