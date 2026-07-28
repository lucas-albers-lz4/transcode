# Documentation Overhaul Plan — transcode

## Current state

transcode has ~890 lines of docs across 9 files. The docs are functional — a new user can install and run the tool — but have accumulated documentation debt:

- **README.md** is overloaded (253 lines) mixing end-user content with developer build instructions
- **docs/DEVELOPMENT.md** (322 lines) is a legacy planning document with stale line-number references, broken links, and ✅ planning markers that blur "planned" vs "shipped"
- No `CONTRIBUTING.md`, `FAQ.md`, `CHANGELOG.md`, `CODE_OF_CONDUCT.md`, or `SECURITY.md`
- One broken link (`yourusername` placeholder in README clone URL)
- Two dead file references (non-existent `ruff-all-stats.txt` / `ruff-all-baseline.json`)

No stale org references exist in code or scripts. No formal GitHub releases exist yet. No release tags.

## Stacked PRs

### Stack 1 — Accuracy & Quick Wins

**Scope:** Fix factual errors and dead links. Touch only existing files.

- `README.md`: Replace `yourusername` with `lucas-albers-lz4` in the clone URL
- `docs/DEVELOPMENT.md`:
  - Remove broken references to `docs/ruff-all-stats.txt` and `docs/ruff-all-baseline.json` (line 127)
  - Strip the "Notes related to current code" section (lines 238–278) — stale line-number references to a long-refactored codebase

**Dependency:** None. Ships first.

### Stack 2 — Community Files

**Scope:** Add the two most impactful missing community files.

- Create `CONTRIBUTING.md`:
  - Architecture overview (component responsibilities from DEVELOPMENT.md)
  - Dev setup (clone → venv → install deps)
  - Linting commands (`ruff check .`, `ruff format .`)
  - Testing (`pytest tests/ -q`)
  - Pre-commit setup
  - Build instructions (PyInstaller / packaging)
- Create `FAQ.md`:
  - "Why is my output file bigger than the input?"
  - "How do I check if FFmpeg has NVENC support?"
  - "How do I fix 'TclError: no display' on Linux?"
  - "The app says FFmpeg is missing — I just installed it"
  - "What's the difference between Archive, Fast, and Quality profiles?"
  - "Can I convert subtitles?"

**Dependency:** None on Stack 1 (different files). Merges after Stack 1.

### Stack 3 — README Split

**Scope:** Restructure README as a quick-start entry point, extract detailed content into dedicated docs.

- Rewrite `README.md` as a streamlined quick-start:
  - One-paragraph project description
  - Requirements (Python 3.8+, FFmpeg)
  - Installation (clone → venv → deps) with cross-links
  - Quick-start: GUI wizard (recommended) and CLI one-liner
  - Build from source (brief, points to `docs/building.md`)
  - Troubleshooting (brief, points to `FAQ.md`)
  - License + links
- Create `docs/user-guide.md`:
  - Full CLI reference (every flag with description)
  - Profile comparison table
  - Examples section
  - GUI wizard flow (Step 1 → Step 2)
  - Advanced: manifest mode, benchmarking, analysis-only, dry-run
- Create `docs/building.md`:
  - PyInstaller build (`./scripts/build_gui.sh`)
  - Release packaging (`./scripts/package_release.sh`)
  - Cross-platform build notes (must build on each target OS)
  - macOS code signing / Gatekeeper notes

**Dependency:** References `CONTRIBUTING.md` and `FAQ.md` from Stack 2. Merges after Stack 2.

### Stack 4 — DEVELOPMENT.md Rewrite

**Scope:** Convert the legacy planning document into a proper developer guide.

- Rewrite `docs/DEVELOPMENT.md`:
  - Remove all "Development Plan" framing, ✅ markers, "Pending Tasks", "Future Enhancements", "Notes related to current code"
  - Keep and refresh: architecture overview, component responsibility table (`scan_media.py`, `convert_media.py`, `convert_to_h265.py`, `media_analysis.py`, `ffmpeg_utils.py`, `analyze_space.py`, `analyze_errors.py`, `workflow.py`, GUI modules)
  - Keep and refresh: linting section (ruff rules, CI target, pre-commit)
  - Keep and refresh: design decisions (file handling, resumability, audio handling, hardware fallback)
  - Cross-reference `CONTRIBUTING.md` for dev setup / build steps
  - Add: GUI architecture overview (`transcode_gui.py`, `gui/` module breakdown)
  - Add: test organization reference (which tests cover which components)

**Dependency:** Cross-references `CONTRIBUTING.md` from Stack 2. Merges after Stack 3.

### Merge order

**Stack 1 → Stack 2 → Stack 3 → Stack 4.** Each branch is based on the previous one. No parallel stacking to avoid rebase conflicts across file rewrites.

## File summary (before → after)

| File | Before | After | Notes |
|------|--------|-------|-------|
| `README.md` | 253 lines | ~120 lines | Trimmed to quick-start |
| `docs/DEVELOPMENT.md` | 322 lines | ~180 lines | Rewritten as dev guide |
| `docs/ENCODING.md` | 77 lines | 77 lines | Unchanged |
| `docs/MANUAL_QA_GUI.md` | 35 lines | 35 lines | Unchanged |
| `docs/TODO.md` | 34 lines | 34 lines | Unchanged |
| `packaging/INSTALL.txt` | 48 lines | 48 lines | Unchanged |
| `CONTRIBUTING.md` | — | ~100 lines | New |
| `FAQ.md` | — | ~80 lines | New |
| `docs/user-guide.md` | — | ~180 lines | New |
| `docs/building.md` | — | ~60 lines | New |

## Acceptance criteria

- [ ] `rg "yourusername" README.md docs/` returns no hits
- [ ] `CONTRIBUTING.md` contains working lint/test commands a new contributor can run
- [ ] `FAQ.md` addresses at least the 6 listed questions
- [ ] Every cross-reference between doc files resolves correctly (no dead relative links)
- [ ] After all stacks merged, `rg "broken|stale|todo" docs/DEVELOPMENT.md` shows only intentional mentions, not planning markers
- [ ] README's quick-start section is under 150 lines
