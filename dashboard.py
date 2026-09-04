"""AegisStore — Risk-Adaptive AI Storage Optimizer · Streamlit Dashboard"""
import shutil, time
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aegisstore import (
    archaeology, context, db, decision_engine, executor,
    future_usage_model, ml_training, predictor, recommendation_engine,
    safety_gate, scanner, storage_intelligence, storage_story, usage_intelligence,
)
from demo_setup import build_demo

st.set_page_config(page_title="AegisStore", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")
db.init_db()
DEFAULT_TARGET = Path("./demo_disk")

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── BASE ── */
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
.stApp { background: #050b1a !important; }
section[data-testid="stSidebar"] { background:#070d1f !important; }
.block-container { padding:1rem 2rem 3rem 2rem !important; max-width:1400px !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#0a1020; }
::-webkit-scrollbar-thumb { background:#1a3a5c; border-radius:3px; }

/* ── STREAMLIT DEFAULTS ── */
.stButton>button {
  background:linear-gradient(135deg,#0d3b6e,#1a5fa8) !important;
  color:#e0f0ff !important; border:1px solid #1e6abf !important;
  border-radius:8px !important; font-weight:600 !important;
  font-family:'Inter',sans-serif !important; font-size:0.85rem !important;
  padding:0.45rem 1.2rem !important; transition:all 0.2s !important;
  box-shadow:0 0 12px rgba(26,95,168,0.25) !important;
}
.stButton>button:hover {
  background:linear-gradient(135deg,#1a5fa8,#2a7fd4) !important;
  box-shadow:0 0 20px rgba(42,127,212,0.45) !important;
  transform:translateY(-1px) !important;
}
.stButton>button[kind="primary"] {
  background:linear-gradient(135deg,#1e3a8a,#2563eb,#3b82f6) !important;
  border:1px solid #3b82f6 !important;
  box-shadow:0 0 24px rgba(59,130,246,0.4),0 0 48px rgba(59,130,246,0.1) !important;
}
.stTextInput>div>div>input {
  background:#0a1428 !important; border:1px solid #1a3a5c !important;
  color:#a0c4e8 !important; border-radius:8px !important; font-family:'JetBrains Mono',monospace !important;
}
.stTextInput>div>div>input:focus { border-color:#2a7fd4 !important; box-shadow:0 0 0 2px rgba(42,127,212,0.2) !important; }
.stSelectbox>div>div { background:#0a1428 !important; border:1px solid #1a3a5c !important; border-radius:8px !important; color:#a0c4e8 !important; }
.stExpander { background:#070e1e !important; border:1px solid #1a2a40 !important; border-radius:10px !important; }
.stExpander summary { color:#7aa8cc !important; font-weight:500 !important; }
.stDataFrame, .stDataFrame table { background:#07101f !important; color:#a0c4e8 !important; }
.stDataFrame th { background:#0d1f3c !important; color:#4aa0e8 !important; font-family:'JetBrains Mono',monospace !important; font-size:0.75rem !important; }
div[data-testid="stMetricValue"] { color:#58a6ff !important; font-size:1.5rem !important; font-family:'JetBrains Mono',monospace !important; }
div[data-testid="stMetricLabel"] { color:#5a7a9a !important; font-size:0.75rem !important; text-transform:uppercase !important; letter-spacing:0.8px !important; }
.stProgress > div > div { background:linear-gradient(90deg,#1a5fa8,#2a7fd4,#3b9ae0) !important; border-radius:999px !important; }
.stSuccess { background:rgba(30,140,74,0.15) !important; border:1px solid rgba(30,140,74,0.3) !important; border-radius:8px !important; color:#4ade80 !important; }
.stInfo { background:rgba(26,95,168,0.12) !important; border:1px solid rgba(26,95,168,0.3) !important; border-radius:8px !important; color:#7dd3fc !important; }
.stWarning { background:rgba(176,125,18,0.12) !important; border:1px solid rgba(176,125,18,0.3) !important; border-radius:8px !important; color:#fcd34d !important; }
.stError { background:rgba(166,50,40,0.12) !important; border:1px solid rgba(166,50,40,0.3) !important; border-radius:8px !important; color:#fca5a5 !important; }
hr { border-color:#0f2040 !important; }
p, li { color:#8ba8c8 !important; }
.stCaption { color:#4a6a8a !important; font-size:0.75rem !important; }
label { color:#6a90b0 !important; }

/* ── CUSTOM COMPONENTS ── */
.hero-wrap {
  background:linear-gradient(135deg,#050e24 0%,#071428 40%,#090f2a 100%);
  border:1px solid rgba(42,127,212,0.2);
  border-radius:18px; padding:2rem 2.2rem 1.6rem;
  margin-bottom:1.4rem; position:relative; overflow:hidden;
}
.hero-wrap::before {
  content:''; position:absolute; top:-80px; right:-80px;
  width:300px; height:300px;
  background:radial-gradient(circle,rgba(59,130,246,0.08) 0%,transparent 70%);
  border-radius:50%;
}
.hero-wrap::after {
  content:''; position:absolute; bottom:-60px; left:30%;
  width:200px; height:200px;
  background:radial-gradient(circle,rgba(139,92,246,0.06) 0%,transparent 70%);
  border-radius:50%;
}
.hero-logo {
  font-family:'Orbitron',monospace; font-size:2rem; font-weight:800;
  background:linear-gradient(90deg,#3b82f6,#60a5fa,#93c5fd);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  letter-spacing:1px; margin:0 0 0.3rem;
}
.hero-tag {
  display:inline-block; background:rgba(59,130,246,0.1);
  border:1px solid rgba(59,130,246,0.25); color:#60a5fa;
  font-size:0.68rem; font-family:'JetBrains Mono',monospace;
  padding:2px 12px; border-radius:20px; margin-bottom:0.6rem;
  letter-spacing:1px; text-transform:uppercase;
}
.hero-sub { color:#4a7090 !important; font-size:0.9rem; max-width:640px; }

.kpi-card {
  background:linear-gradient(135deg,#060d1f,#091526);
  border:1px solid #0f2540; border-radius:14px;
  padding:1.1rem 1rem; text-align:center; position:relative; overflow:hidden;
  transition:border-color 0.2s, box-shadow 0.2s;
}
.kpi-card:hover { border-color:#1a5fa8; box-shadow:0 0 20px rgba(26,95,168,0.15); }
.kpi-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,transparent,#1a5fa8,transparent);
}
.kpi-label { font-size:0.68rem; color:#3a6080; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem; }
.kpi-value { font-size:1.8rem; font-weight:700; font-family:'JetBrains Mono',monospace; color:#58a6ff; line-height:1; }
.kpi-value.green { color:#22c55e; }
.kpi-value.amber { color:#f59e0b; }
.kpi-value.red   { color:#ef4444; }
.kpi-value.teal  { color:#14b8a6; }
.kpi-value.purple{ color:#a78bfa; }
.kpi-sub { font-size:0.7rem; color:#2a5070; margin-top:0.3rem; }

.live-card {
  background:#060c1a; border:1px solid #0d1e35; border-radius:12px;
  padding:0.8rem 0.7rem; text-align:center;
}
.live-label { font-size:0.65rem; color:#2a5070; text-transform:uppercase; letter-spacing:0.8px; }
.live-val { font-size:1.2rem; font-weight:600; font-family:'JetBrains Mono',monospace; color:#a0c4e8; }

.sec-head {
  display:flex; align-items:center; gap:0.6rem;
  font-size:0.85rem; font-weight:600; text-transform:uppercase;
  letter-spacing:1.2px; color:#3a7aaa;
  margin:1.8rem 0 0.9rem; padding-bottom:0.5rem;
  border-bottom:1px solid #0a1e35;
}
.sec-head .dot {
  width:8px; height:8px; border-radius:50%;
  background:#1a5fa8; box-shadow:0 0 8px #1a5fa8;
  flex-shrink:0;
}

.badge {
  display:inline-block; padding:2px 10px; border-radius:999px;
  font-size:0.7rem; font-weight:700; font-family:'JetBrains Mono',monospace;
  letter-spacing:0.5px;
}
.badge-low  { background:rgba(34,197,94,0.12);  color:#22c55e; border:1px solid rgba(34,197,94,0.25); }
.badge-med  { background:rgba(245,158,11,0.12); color:#f59e0b; border:1px solid rgba(245,158,11,0.25); }
.badge-high { background:rgba(239,68,68,0.12);  color:#ef4444; border:1px solid rgba(239,68,68,0.25); }

.arc-card {
  background:linear-gradient(135deg,rgba(26,95,168,0.08),rgba(139,92,246,0.06));
  border:1px solid rgba(26,95,168,0.2); border-left:4px solid #1a5fa8;
  border-radius:10px; padding:0.85rem 1.1rem; margin-bottom:0.7rem;
}
.arc-card .ah { font-weight:600; color:#a0c4e8; font-size:0.9rem; }
.arc-card .ad { color:#3a6080; font-size:0.8rem; margin-top:0.2rem; }

.safety-warn {
  background:linear-gradient(135deg,rgba(239,68,68,0.08),rgba(239,68,68,0.04));
  border:1px solid rgba(239,68,68,0.3); border-left:4px solid #ef4444;
  border-radius:10px; padding:0.8rem 1.1rem; color:#ef4444;
  font-weight:500; font-size:0.88rem;
}
.factor-pill {
  display:inline-block; background:rgba(26,95,168,0.1);
  border:1px solid rgba(26,95,168,0.2); border-radius:6px;
  padding:2px 9px; font-size:0.75rem; color:#4a80aa;
  margin:2px 2px 2px 0; font-family:'JetBrains Mono',monospace;
}
.cf-card {
  background:#060c1a; border:1px solid #0d1e35; border-radius:12px;
  padding:1rem 1.2rem;
}
.timeline-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ────────────────────────────────────────────────────────────────────
def ensure_demo(target):
    if not target.exists():
        build_demo(target)
    total, used, _ = shutil.disk_usage(target)
    db.log_usage(str(target), used, total)
    if predictor.forecast(str(target), min_points=3) is None:
        predictor.seed_synthetic_history(str(target), total, current_used_bytes=used,
                                          daily_growth_gb=1.8, days_back=14)

if "boot" not in st.session_state:
    ensure_demo(DEFAULT_TARGET)
    st.session_state.boot = True

@st.cache_resource(show_spinner=False)
def trained_model():
    X, y = ml_training.generate_training_data(samples=3000, seed=42)
    return future_usage_model.train_model(X, y)

def hb(b):
    for u in ["B","KB","MB","GB","TB"]:
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} PB"

def badge(tier):
    c = {"LOW":"badge-low","MEDIUM":"badge-med","HIGH":"badge-high"}.get(tier,"badge-med")
    return f'<span class="badge {c}">{tier}</span>'

def section(icon, label):
    st.markdown(f'<div class="sec-head"><span class="dot"></span>{icon} {label}</div>', unsafe_allow_html=True)

def pie(labels, values, colors, title):
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.6,
        marker=dict(colors=colors, line=dict(color="#050b1a", width=4)),
        textinfo="percent",
        textfont=dict(size=12, family="JetBrains Mono", color="#c0d8f0"),
        hovertemplate="<b>%{label}</b><br>%{value} · %{percent}<extra></extra>",
        pull=[0.05]*len(labels),
    ))
    fig.update_layout(
        annotations=[dict(text=title, x=0.5, y=0.5, font_size=11,
                          font_color="#3a6080", font_family="Inter", showarrow=False)],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6a90b0", family="Inter"),
        legend=dict(font=dict(size=10, color="#4a7090"), bgcolor="rgba(0,0,0,0)",
                    orientation="v", x=1.02, xanchor="left"),
        margin=dict(t=10, b=10, l=10, r=10), height=260, showlegend=True,
    )
    return fig


# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-tag">Track 2 · AI at Application Level · C-DAC Hackathon · Team Linux_Squad</div>
  <div class="hero-logo">🛡 AEGISSTORE</div>
  <p class="hero-sub">AI understands what can be optimized.<br>
  AegisStore decides whether it is safe to act — and learns from every decision you accept or reject.</p>
</div>
""", unsafe_allow_html=True)

# ── CONTROLS ───────────────────────────────────────────────────────────────────
ci, cs, cr = st.columns([3,1,1])
target_dir = ci.text_input("dir", value="./demo_disk", label_visibility="collapsed")
ci.caption("📁 Target directory")
scan_clicked = cs.button("🔍  Scan Now", use_container_width=True, type="primary")
reset_clicked = cr.button("♻️  Reset", use_container_width=True, help="Wipe and rebuild demo_disk")

with st.expander("🛡️ Safety Guarantees & Threat Model"):
    st.markdown("""
| Guarantee | Mechanism |
|---|---|
| No direct deletion | All cleanup routed through quarantine with SHA-256 |
| Open file protection | Active-process files are blocked at safety gate |
| Risk-threshold gating | Every file scored 0–100 before any action |
| Live-load deferral | High CPU / RAM / IO-wait defers all cleanup in real time |
| Dependency awareness | Package-owned, Git-tracked, symlinks — all flagged |
| Feedback calibration | Accept / reject clicks retrain risk boundaries via logistic regression |
    """)

if reset_clicked:
    t = Path(target_dir)
    if t.exists(): shutil.rmtree(t)
    for p in [Path("quarantine"), Path("aegisstore.db"), Path("calibration.json")]:
        if p.exists(): (shutil.rmtree(p) if p.is_dir() else p.unlink())
    for k in list(st.session_state.keys()): del st.session_state[k]
    st.rerun()

# ── LIVE SYSTEM LOAD ───────────────────────────────────────────────────────────
load = safety_gate.read_system_load(sample_seconds=0.3)
busy = safety_gate.is_system_busy(load)
state = str(load.get("state","NORMAL"))
sc = {"NORMAL":"#22c55e","BUSY":"#f59e0b","CRITICAL":"#ef4444"}.get(state,"#6a90b0")

section("💻", "Live System Monitor")
lc = st.columns(6)
for col, lbl, val, unit in [
    (lc[0],"CPU",     f"{load['cpu_percent']:.0f}",        "%"),
    (lc[1],"RAM",     f"{load['memory_percent']:.0f}",     "%"),
    (lc[2],"Disk R",  f"{load['disk_read_mb_s']:.1f}",     " MB/s"),
    (lc[3],"Disk W",  f"{load['disk_write_mb_s']:.1f}",    " MB/s"),
    (lc[4],"IO Wait", f"{load['io_wait_percent']:.0f}",    "%"),
]:
    col.markdown(f"""<div class="live-card">
    <div class="live-label">{lbl}</div>
    <div class="live-val">{val}<span style="font-size:.65rem;color:#2a5070">{unit}</span></div>
    </div>""", unsafe_allow_html=True)

lc[5].markdown(f"""<div class="live-card">
<div class="live-label">Safety Gate</div>
<div class="live-val" style="color:{sc};text-shadow:0 0 8px {sc}88">{state}</div>
</div>""", unsafe_allow_html=True)

if busy:
    st.markdown(f"""<div class="safety-warn">⚡ LIVE OVERRIDE ACTIVE — cleanup deferred.
    CPU {load['cpu_percent']:.0f}% · RAM {load['memory_percent']:.0f}% · IO {load['io_wait_percent']:.0f}%</div>""",
    unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = None; st.session_state.summary = None; st.session_state.reclaimable = 0

# ── SCAN PIPELINE ──────────────────────────────────────────────────────────────
if scan_clicked:
    target = str(Path(target_dir))
    if not Path(target).exists():
        st.error(f"'{target}' does not exist.")
    else:
        prog = st.progress(0); status = st.empty()

        status.markdown("**📁 Scanning filesystem...**"); records = scanner.scan_and_classify(target)
        total, used, _ = shutil.disk_usage(target); db.log_usage(target, used, total); prog.progress(15)

        status.markdown("**📊 Analyzing usage intelligence...**")
        analyzed = usage_intelligence.analyze_records(records); prog.progress(30)

        status.markdown("**🤖 Predicting future usage (ML)...**")
        model = trained_model()
        predicted = [future_usage_model.predict_record(model, r) for r in analyzed]; prog.progress(50)

        status.markdown("**⚙️ Computing risk-adaptive decisions...**")
        candidates, reclaimable = scanner.reclaimable_summary(records)
        cpaths = {str(c["path"]) for c in candidates}
        crecs = sorted([r for r in predicted if str(r["path"]) in cpaths],
                       key=lambda r: r["size_bytes"], reverse=True)[:20]
        prog.progress(65)

        rows = []
        for r in crecs:
            ctx = context.enrich(str(r["path"])); merged = {**r, **ctx}
            dec = decision_engine.assess(r, ctx, load, busy)
            rec = recommendation_engine.recommend(merged)
            if dec["action"] == "DEFER":
                db.log_schedule_event(r["path"], "DEFERRED", load, reason=dec["reason"])
            cid = db.save_candidate(r)
            db.save_decision(cid, {**ctx, "cpu_percent": load["cpu_percent"],
                                    "io_wait_percent": load["io_wait_percent"], **dec})
            rows.append({
                "File": r["path"].name, "Path": str(r["path"]),
                "Size": hb(r["size_bytes"]), "size_bytes": r["size_bytes"],
                "Age (days)": r["age_days"], "Classification": r["classification"],
                "Confidence": f"{r['confidence']:.0%}",
                "risk_score": dec["risk_score"], "risk_tier": dec["risk_tier"],
                "action": dec["action"], "reason": dec["reason"],
                "factors": dec.get("factors", []),
                "Risk": dec["risk_tier"], "Risk Score": f"{dec['risk_score']} / 100",
                "Action": dec["action"], "Reason": dec["reason"],
                "usage_profile": r.get("usage_profile"),
                "future_usage_probability": r.get("future_usage_probability"),
                "future_usage_class": r.get("future_usage_class"),
                "future_usage_explanation": r.get("future_usage_explanation"),
                "recommendation": rec.get("recommendation"),
                "recommendation_reason": rec.get("recommendation_reason"),
            })
        prog.progress(85)
        status.markdown("**📈 Forecasting & generating stories...**")

        st.session_state.results = rows; st.session_state.archaeology_records = analyzed
        st.session_state.reclaimable = reclaimable; st.session_state.total_disk = total
        st.session_state.used_disk = used; st.session_state.target = target

        auto = [r for r in rows if r["Action"] == "AUTOMATE"]
        fc = predictor.forecast(target)
        fc_det = storage_intelligence.storage_forecast_detailed(target, reclaimable_bytes=reclaimable)
        summary = {"total_candidates": len(records), "total_reclaimable_gb": reclaimable/(1024**3),
                   "top_reason": "cold/redundant data", "deferred_count": len([r for r in rows if r["Action"]=="DEFER"]),
                   "automated_count": len(auto), "avg_confidence": 0.85}
        if fc: summary["growth_rate_gb_per_day"] = fc["growth_rate_gb_per_day"]
        st.session_state.forecast = fc; st.session_state.forecast_detailed = fc_det
        st.session_state.summary = summary

        prog.progress(100); status.markdown(f"**✅ Scan complete — {len(rows)} candidates found.**")
        time.sleep(0.4); prog.empty(); status.empty()

# ── RESULTS ────────────────────────────────────────────────────────────────────
if st.session_state.results is not None:
    used = st.session_state.used_disk; total = st.session_state.total_disk
    results = st.session_state.results

    section("📊","Scan Summary")
    km = st.columns(4)
    km[0].markdown(f"""<div class="kpi-card">
    <div class="kpi-label">Disk Usage</div>
    <div class="kpi-value amber">{used/total:.0%}</div>
    <div class="kpi-sub">{hb(used)} / {hb(total)}</div>
    </div>""", unsafe_allow_html=True)
    km[1].markdown(f"""<div class="kpi-card">
    <div class="kpi-label">Reclaimable</div>
    <div class="kpi-value green">{hb(st.session_state.reclaimable)}</div>
    <div class="kpi-sub">Safe to recover</div>
    </div>""", unsafe_allow_html=True)
    km[2].markdown(f"""<div class="kpi-card">
    <div class="kpi-label">Candidates</div>
    <div class="kpi-value">{len(results)}</div>
    <div class="kpi-sub">Files flagged</div>
    </div>""", unsafe_allow_html=True)
    km[3].markdown(f"""<div class="kpi-card">
    <div class="kpi-label">Ready to Clean</div>
    <div class="kpi-value teal">{sum(1 for r in results if r["Action"]=="AUTOMATE")}</div>
    <div class="kpi-sub">AUTOMATE eligible</div>
    </div>""", unsafe_allow_html=True)

    # ── PIE CHARTS ──────────────────────────────────────────────────────────────
    section("🥧","Visual Breakdown")
    pc1, pc2, pc3 = st.columns(3)

    risk_c = {t:sum(1 for r in results if r.get("risk_tier")==t) for t in ["LOW","MEDIUM","HIGH"]}
    with pc1:
        if any(risk_c.values()):
            st.plotly_chart(pie(list(risk_c.keys()), list(risk_c.values()),
                ["#15803d","#b45309","#b91c1c"], "Risk Tier"), use_container_width=True,
                config={"displayModeBar":False})

    clf_c = {}
    for r in results:
        c = r.get("Classification","Unknown"); clf_c[c] = clf_c.get(c,0)+1
    clf_col = {"HOT":"#b91c1c","WARM":"#b45309","COLD":"#1d4ed8","REDUNDANT":"#6d28d9","Unknown":"#1e3a5f","Cold + Redundant":"#0f766e"}
    with pc2:
        if clf_c:
            st.plotly_chart(pie(list(clf_c.keys()), list(clf_c.values()),
                [clf_col.get(k,"#1e3a5f") for k in clf_c], "File Classification"),
                use_container_width=True, config={"displayModeBar":False})

    rec_c = {}
    for r in results:
        a = r.get("recommendation") or r.get("Action","—"); rec_c[a] = rec_c.get(a,0)+1
    rec_col = {"CLEANUP":"#b91c1c","ARCHIVE":"#b45309","KEEP":"#15803d","REVIEW":"#1d4ed8",
               "AUTOMATE":"#0f766e","DEFER":"#6d28d9","SKIP":"#1e3a5f","APPROVAL_REQUIRED":"#92400e"}
    with pc3:
        if rec_c:
            st.plotly_chart(pie(list(rec_c.keys()), list(rec_c.values()),
                [rec_col.get(k,"#1e3a5f") for k in rec_c], "Recommended Actions"),
                use_container_width=True, config={"displayModeBar":False})

    thr = decision_engine.current_thresholds()
    st.caption(f"Risk boundary: LOW < {thr['low_threshold']} · HIGH ≥ {thr['high_threshold']} · "
               f"{'default' if thr['is_default'] else '🔄 recalibrated from feedback'}")

    # ── FORECAST ────────────────────────────────────────────────────────────────
    fc_det = st.session_state.get("forecast_detailed")
    if fc_det:
        fc = fc_det["forecast"]
        section("📈","Growth Forecast")
        if fc:
            fcols = st.columns(5)
            fcols[0].metric("Usage Now",   f"{fc['current_usage_pct']*100:.0f}%")
            fcols[1].metric("Growth/Day",  fc_det["growth_rate_formatted"])
            d85 = fc["predictions_days"].get(0.85)
            fcols[2].metric("Days → 85%",  f"{d85:.0f}" if d85 else "N/A")
            d90 = fc["predictions_days"].get(0.90)
            fcols[3].metric("Days → 90%",  f"{d90:.0f}" if d90 else "N/A")
            d95 = fc["predictions_days"].get(0.95)
            fcols[4].metric("Days → 95%",  f"{d95:.0f}" if d95 else "N/A")

            hist = db.usage_series(st.session_state.target)
            if len(hist) >= 2:
                cdf = pd.DataFrame([{"Date": datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d"),
                                      "Usage (GB)": r["used_bytes"]/(1024**3)} for r in hist])
                cdf = cdf.drop_duplicates("Date",keep="last").set_index("Date")

                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=cdf.index, y=cdf["Usage (GB)"],
                    mode="lines+markers",
                    line=dict(color="#2563eb", width=2.5, shape="spline"),
                    marker=dict(color="#3b82f6", size=6,
                                line=dict(color="#1d4ed8", width=1.5)),
                    fill="tozeroy",
                    fillcolor="rgba(37,99,235,0.08)",
                    hovertemplate="<b>%{x}</b><br>%{y:.2f} GB<extra></extra>",
                ))
                fig_line.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#4a7090", family="Inter"),
                    xaxis=dict(gridcolor="#0a1e35", linecolor="#0a1e35", tickfont=dict(size=10)),
                    yaxis=dict(gridcolor="#0a1e35", linecolor="#0a1e35", title="GB",
                               tickfont=dict(size=10)),
                    margin=dict(t=10, b=30, l=50, r=10), height=180,
                )
                st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar":False})

        section("💾","Cleanup Impact Projection")
        ci2 = fc_det.get("cleanup_impact")
        if ci2:
            cic = st.columns(4)
            cic[0].metric("Current",     ci2["current_used_formatted"])
            cic[1].metric("Reclaimable", ci2["reclaimable_formatted"])
            cic[2].metric("After Cleanup", ci2["estimated_after_formatted"])
            cic[3].metric("After %",     f"{ci2['estimated_after_pct']*100:.0f}%")

        section("🧭","Storage Intelligence")
        st.info(fc_det["recommendation"])

    # ── DIGITAL ARCHAEOLOGY ──────────────────────────────────────────────────
    section("🏛️","Digital Archaeology — Storage Story")
    st.caption("Your disk, told as a story a non-technical judge can understand in one glance.")
    arch_recs = st.session_state.get("archaeology_records", [])
    stories = archaeology.build_stories(arch_recs) if arch_recs else []
    if stories:
        for s in stories[:6]:
            st.markdown(f"""<div class="arc-card">
            <div class="ah">🗂 {s['headline']}</div>
            <div class="ad">{s['detail']}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No large groupings to narrate yet.")

# ── CANDIDATE TABLE ────────────────────────────────────────────────────────────
section("📋","Risk-Adaptive Candidate Results")
results = st.session_state.get("results") or []
df = pd.DataFrame(results)

if not df.empty:
    disp = pd.DataFrame([{
        "File": r["File"], "Size": r["Size"], "Age (d)": r["Age (days)"],
        "Class": r["Classification"],
        "Profile": r.get("usage_profile") or "—",
        "Future Use": (f"{r['future_usage_probability']*100:.1f}%"
                       if r.get("future_usage_probability") is not None else "—"),
        "Score": r["Risk Score"], "Risk": r["Risk"],
        "Action": r.get("recommendation") or r["Action"],
        "Reason": r.get("recommendation_reason") or r["Reason"],
    } for r in results])

    def rc(v):
        return {"LOW":"background-color:#0d2a15;color:#22c55e",
                "MEDIUM":"background-color:#2a1e00;color:#f59e0b",
                "HIGH":"background-color:#2a0a0a;color:#ef4444"}.get(v,"")

    st.dataframe(disp.style.map(rc, subset=["Risk"]), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Export CSV",
        data=disp.to_csv(index=False).encode(),
        file_name=f"aegisstore_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

    # ── WHY THIS DECISION ────────────────────────────────────────────────────
    section("🔍","Decision Inspector")
    sel_name = st.selectbox("Select candidate", [r["File"] for r in results], index=0)
    sel = next((r for r in results if r["File"] == sel_name), None)
    if sel:
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Risk Score", sel["Risk Score"])
        dc2.metric("Action",     sel["Action"])
        dc3.metric("Age",        f"{sel['Age (days)']} days")

        st.markdown(f"**File:** `{sel['File']}`  {badge(sel['Risk'])}", unsafe_allow_html=True)
        if sel.get("recommendation"):
            st.markdown(f"**ML Recommendation:** `{sel['recommendation']}` — {sel.get('recommendation_reason','')}")
        if sel.get("future_usage_probability") is not None:
            p = sel['future_usage_probability']*100
            bar_col = "#22c55e" if p<30 else "#f59e0b" if p<60 else "#ef4444"
            st.markdown(f"""
            <div class="cf-card">
              <div style="font-size:.75rem;color:#3a6080;margin-bottom:.4rem">PREDICTED FUTURE USE</div>
              <div style="font-size:1.6rem;font-family:'JetBrains Mono';color:{bar_col};font-weight:700">{p:.1f}%</div>
              <div style="background:#0a1428;border-radius:999px;height:6px;margin:.5rem 0">
                <div style="background:{bar_col};width:{min(p,100):.0f}%;height:6px;border-radius:999px;box-shadow:0 0 6px {bar_col}88"></div>
              </div>
              <div style="font-size:.78rem;color:#3a6080">{sel.get('future_usage_class','—')} — {sel.get('future_usage_explanation','')}</div>
            </div>
            """, unsafe_allow_html=True)

        factors = sel.get("factors", [])
        if factors:
            st.markdown("**Scoring Factors:**")
            st.markdown("".join(f'<span class="factor-pill">{f}</span>' for f in factors),
                       unsafe_allow_html=True)
        st.markdown(f"**Decision:** {sel['Reason']}")

        # Counterfactual
        st.markdown("---")
        section("🔄","Counterfactual — What Would Change This Decision?")
        try:
            from aegisstore import counterfactual
            orig = next((c for c in scanner.scan_and_classify(st.session_state.target)
                         if str(c["path"]) == sel["Path"]), None)
            if orig:
                cf = counterfactual.explain_age_change(
                    orig, context.enrich(str(orig["path"])), load, busy, days_delta=-7)
                cfc1, cfc2 = st.columns(2)
                cfc1.metric("Current Risk", f"{cf['current_score']} / 100")
                cfc2.metric("If 7 Days Newer", f"{cf['counterfactual_score']} / 100",
                            delta=cf["delta"])
                st.info(f"💡 {cf['explanation']}")
                if cf["current_action"] != cf["counterfactual_action"]:
                    st.warning(f"Decision changes: **{cf['current_action']}** → **{cf['counterfactual_action']}**")
        except Exception as e:
            st.caption(f"Counterfactual: {e}")

        # Feedback
        st.markdown("---")
        section("🎓","Was This Recommendation Right?")
        st.caption("Your feedback retrains risk-tier thresholds via logistic regression.")
        fb1, fb2, fb3 = st.columns([1,1,3])
        rec_lbl = sel.get("recommendation") or sel["Action"]
        if fb1.button("👍 Accept", key=f"a_{sel['Path']}"):
            db.log_recommendation_feedback(sel["Path"], rec_lbl, sel["risk_score"],
                                            sel.get("future_usage_probability"), accepted=True)
            st.success("Feedback logged: accepted ✓")
        if fb2.button("👎 Reject", key=f"r_{sel['Path']}"):
            db.log_recommendation_feedback(sel["Path"], rec_lbl, sel["risk_score"],
                                            sel.get("future_usage_probability"), accepted=False)
            st.success("Feedback logged: rejected ✓")
        fb3.caption(f"{db.recommendation_feedback_count()} samples (need 6+ to recalibrate)")

    section("🧠","Recalibrate Risk Thresholds")
    rc1, rc2 = st.columns([1,3])
    if rc1.button("Recalibrate Now"):
        import recalibrate as rm
        r = rm.recalibrate_from_feedback()
        if r["status"] == "recalibrated":
            st.success(f"Done! LOW: {r['old_thresholds']['low_threshold']} → {r['new_thresholds']['low_threshold']} · "
                       f"HIGH: {r['old_thresholds']['high_threshold']} → {r['new_thresholds']['high_threshold']}")
        elif r["status"] == "insufficient_data":
            st.info(f"Only {r['sample_count']} samples — need 6+.")
        else:
            st.info(f"Status: {r.get('error', r['status'])}")
    if rc2.button("Reset to Defaults"):
        import recalibrate as rm; rm.reset_calibration()
        st.success("Reset to LOW < 31, HIGH ≥ 66")

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#1a3a5c;font-family:'Orbitron',monospace;font-size:0.9rem;letter-spacing:2px">
    ◆ AWAITING SCAN — PRESS SCAN NOW TO ANALYZE ◆
    </div>""", unsafe_allow_html=True)

# ── TAKE ACTION ────────────────────────────────────────────────────────────────
if results:
    section("⚡","Take Action")
    auto_el  = [r for r in results if r["Action"]=="AUTOMATE"]
    sched    = [r for r in results if r["Action"]=="SCHEDULE"]
    approval = [r for r in results if r["Action"]=="APPROVAL_REQUIRED"]
    deferred = [r for r in results if r["Action"]=="DEFER"]
    skipped  = [r for r in results if r["Action"]=="SKIP"]

    if auto_el:
        st.success(f"🟢 {len(auto_el)} candidates are safe for automatic cleanup.")
        cb1, cb2 = st.columns([2,1])
        if cb1.button("⚡ Execute Batch Cleanup", type="primary"):
            br = executor.batch_quarantine(
                [{"path":r["Path"],"reason":r["Reason"]} for r in auto_el], load, verify_safety=True)
            if br["safety_cleared"]:
                if br["executed"]:
                    st.success(f"✓ {len(br['executed'])} quarantined · {br['total_bytes_recovered']/(1024**3):.2f} GB recovered")
                    st.balloons()
                if br["failed"]:
                    st.warning(f"⚠ {len(br['failed'])} failed.")
            else:
                st.warning("🛑 Safety gate blocked — system busy.")
        cb2.metric("Batch Safety", "✓ PASS" if not busy else "⚠ DEFER")

    action_map = [("AUTOMATE",auto_el,"LOW"),("SCHEDULE",sched,"MEDIUM"),
                  ("APPROVAL_REQUIRED",approval,"MEDIUM"),("DEFER",deferred,"HIGH"),("SKIP",skipped,"HIGH")]
    for aname, items, tier in action_map:
        if items:
            st.markdown(f"**{aname}** {badge(tier)}", unsafe_allow_html=True)
            for r in items:
                c1,c2 = st.columns([4,1])
                c1.write(f"`{r['File']}` · {r['Size']} · {r['Reason']}")
                if aname=="AUTOMATE":
                    if c2.button("Quarantine", key=f"q_{r['Path']}"):
                        try:
                            info = executor.quarantine_file(r["Path"], r["Reason"])
                            st.success(f"Quarantined ✓ integrity={info['integrity_verified']}")
                        except FileNotFoundError:
                            st.warning("Already quarantined.")
                else:
                    c2.write("—")

    section("📖","Storage Story")
    with st.spinner("Generating narrative..."):
        story = storage_story.generate_story(st.session_state.summary)
    st.info(story)

# ── RECOVERY ───────────────────────────────────────────────────────────────────
st.divider()
section("🧾","Recovery & Quarantine")
rs = executor.recovery_stats()
qc = st.columns(3)
qc[0].markdown(f"""<div class="kpi-card">
<div class="kpi-label">In Quarantine</div>
<div class="kpi-value">{rs['file_count']}</div>
</div>""", unsafe_allow_html=True)
qc[1].markdown(f"""<div class="kpi-card">
<div class="kpi-label">Total Size</div>
<div class="kpi-value amber">{rs['total_bytes']/(1024**3):.2f} <span style="font-size:.8rem">GB</span></div>
</div>""", unsafe_allow_html=True)
qc[2].markdown(f"""<div class="kpi-card">
<div class="kpi-label">Integrity OK</div>
<div class="kpi-value green">{rs['integrity_ok']}/{rs['file_count']}</div>
</div>""", unsafe_allow_html=True)

qi = executor.list_quarantine(limit=50)
if qi:
    st.dataframe(pd.DataFrame([{
        "File": Path(i["original_path"]).name, "Original": i["original_path"],
        "Reason": i["reason"], "Size (GB)": f"{i['size_bytes']/(1024**3):.2f}",
        "SHA-256 ✓": "✓" if i["integrity_verified"] else "✗",
    } for i in qi]), use_container_width=True, hide_index=True)

    sq = st.selectbox("Restore file", [i["quarantine_path"] for i in qi],
                      format_func=lambda p: Path(p).name)
    if sq and st.button("↩️ Restore to Original Location"):
        try:
            res = executor.undo_last(sq); st.success(f"Restored → {res['restored_to']}"); st.rerun()
        except Exception as e:
            st.error(f"Restore failed: {e}")
else:
    st.info("No files in quarantine.")

# ── AUDIT LOG ──────────────────────────────────────────────────────────────────
section("📜","Audit Log")
al = db.recent_audit(limit=15)
if al:
    st.dataframe(pd.DataFrame([dict(r) for r in al])[["event_time","action","path","reversible","detail"]],
                 use_container_width=True, hide_index=True)
else:
    st.caption("No actions yet.")

section("🕐","Performance-Aware Scheduling Timeline")
sa = {"DEFERRED","RETRIED","EXECUTED","QUARANTINE"}
sr = [dict(r) for r in db.recent_audit(limit=50) if r["action"] in sa]
if sr:
    st.dataframe(pd.DataFrame([{
        "Time": datetime.fromtimestamp(r["event_time"]).strftime("%H:%M:%S"),
        "Event": "EXECUTED" if r["action"]=="QUARANTINE" else r["action"],
        "File": r["path"], "Reason": r["detail"],
    } for r in sr]), use_container_width=True, hide_index=True)
else:
    st.info("No scheduling events yet.")
