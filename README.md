# AegisStore

**A Risk-Adaptive Intelligent Storage Optimization Framework for Linux**

> "AI understands what can be optimized. AegisStore decides whether it is safe to act."

Most AI storage tools follow one pattern: find files → get an AI opinion → delete. AegisStore separates **AI recommendation from AI authority**. Every optimization candidate is scored on data context, AI confidence, and — critically — the system's *live* CPU/I-O load, before it's ever allowed to run. A candidate the model is confident about can still be deferred in real time if the machine is busy right now.

**🔗 Live demo:** https://aegisstore-9tssxb85wbgqe3j4z79c9q.streamlit.app
*(Opens with a self-generated demo environment — no setup required to explore it.)*

---

## The core idea

```
Existing tools:        Find large files → Show user → User decides
Typical "AI cleaner":  Find "unnecessary" files → AI recommendation → Delete
AegisStore:            Understand data → Understand context → Predict growth →
                        Evaluate live workload → Score risk → Decide autonomy →
                        Explain the decision → Quarantine → Verify → Audit
```

Nothing is ever hard-deleted. Every action passes through quarantine with SHA-256 integrity verification, a full audit log, and one-click undo.

---

## Modules

All seven modules from the original architecture are implemented, tested, and live in the deployed app — not just diagrammed.

| # | Module | File | What it does |
|---|--------|------|---------------|
| 1 | Filesystem Collector | `aegisstore/scanner.py` | Walks the filesystem, hashes files, detects duplicates, classifies Hot/Warm/Cold/Redundant |
| 2 | Context Intelligence | `aegisstore/context.py` | Checks active-process usage, package ownership (dpkg/rpm), and Git tracking before anything is touched |
| 3 | ML Prediction Engine | `aegisstore/predictor.py` | Linear-regression growth forecasting from historical usage snapshots |
| 4 | Real-Time Safety Gate | `aegisstore/safety_gate.py` | Live CPU / memory / I-O-wait monitoring |
| 5 | Risk-Adaptive Decision Engine | `aegisstore/decision_engine.py` | Scores every candidate 0–100 across data importance, confidence, context, and live load; outputs a risk tier, an action, and human-readable explanatory factors |
| 6 | Safe Execution Engine | `aegisstore/executor.py` | Quarantine (with integrity verification), undo, batch cleanup, and recovery stats |
| 7 | Explainability & Dashboard | `aegisstore/storage_story.py`, `aegisstore/storage_intelligence.py`, `dashboard.py` | Plain-English narrative (LLM-backed, with a template fallback), forecast-quality scoring, cleanup-impact estimation, and the full Streamlit UI |

Supporting infrastructure: `aegisstore/db.py` (SQLite persistence for history, decisions, and the audit log), `aegisstore/cli.py` (command-line interface), and a `tests/` suite covering the decision engine, executor, safety gate, and storage intelligence layer (32 tests, all passing).

---

## What the dashboard shows

- Live CPU / I-O-wait / safety-gate status
- Growth forecast with a usage-history chart and days-to-capacity estimates
- Potential storage impact of cleanup (before/after projection)
- Per-candidate risk breakdown with an explicit **"Why this decision?"** section
- One-click quarantine, or **batch cleanup** of every low-risk candidate at once
- CSV export of the scan report
- Recovery & quarantine management (list, undo)
- Full audit log
- A **Reset demo** button to rebuild a clean environment on demand — useful for letting someone try the app hands-on without touching a terminal

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`anthropic` is optional. Without an `ANTHROPIC_API_KEY` set, the Storage Story narrative falls back to a deterministic template — the app works fully offline.

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # optional
```

## Run it

**Dashboard (recommended):**
```bash
streamlit run dashboard.py
```
On first load it self-generates a demo environment automatically — no manual setup step needed, on your machine or on a deployed link.

**CLI:**
```bash
python3 demo_setup.py ./demo_disk
python3 -m aegisstore.cli scan ./demo_disk
python3 -m aegisstore.cli audit
python3 -m aegisstore.cli undo <quarantine_file_path>
```

**Prove the real-time override** (a LOW-risk file deferring purely because the system is busy right now — the strongest single proof point):
```bash
python3 test_override.py
```

## Tests

```bash
pip install pytest
python3 -m pytest tests/ -v
```
32 tests covering the decision engine, executor (including batch quarantine and recovery stats), safety gate, and storage intelligence layer.

---

## Tech stack

Python · Streamlit · psutil · SQLite · Anthropic Claude API (optional, narrative only — never given filesystem access)

---

## Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for the full Streamlit Community Cloud deployment guide. In short: push to GitHub, connect the repo at share.streamlit.io with `dashboard.py` as the entry point, and it deploys in a few minutes — the app is self-bootstrapping, so the deployed link works immediately with zero manual setup.

---

## Known limitations (worth being upfront about)

- `is_active_process()` checks open file handles via `psutil`, which requires enough permission to see other processes' open files. Works for user-owned processes on most machines; on a restricted host this is a permissions boundary, not a logic gap.
- The growth forecast is a linear regression over logged usage snapshots — solid for a demo, not a substitute for a real seasonal forecasting model in production.
