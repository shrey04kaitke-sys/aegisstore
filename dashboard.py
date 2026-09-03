"""
dashboard.py — AegisStore visual dashboard (Streamlit).
Run with: streamlit run dashboard.py

This is the single-screen demo surface: scan -> risk decision -> ML future-use
prediction -> recommendation -> counterfactual -> accept/reject feedback ->
recalibration -> digital archaeology summary -> batch/individual actions ->
recovery & audit -> scheduling timeline. Every module in aegisstore/ that the
CLI can exercise is reachable from this one page.
"""
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aegisstore import (
    archaeology, context, db, decision_engine, executor, future_usage_model,
    ml_training, predictor, recommendation_engine, safety_gate, scanner,
    storage_intelligence, storage_story, usage_intelligence,
)
from demo_setup import build_demo

st.set_page_config(page_title="AegisStore", page_icon="🛡️", layout="wide")
db.init_db()

DEFAULT_TARGET = Path("./demo_disk")

# ---------------------------------------------------------------------------
# CSS — dark, professional, security-tool aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Hero */
  .aegis-hero {
    padding: 2rem 2rem 1.6rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.2rem;
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 60%, #060e1a 100%);
    border: 1px solid rgba(56, 139, 253, 0.3);
    position: relative;
    overflow: hidden;
  }
  .aegis-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(56,139,253,0.12) 0%, transparent 70%);
    border-radius: 50%;
  }
  .aegis-hero h1 {
    margin: 0 0 0.3rem 0;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #e6edf3;
  }
  .aegis-hero p {
    margin: 0;
    color: #8b949e;
    font-size: 0.95rem;
    max-width: 620px;
  }
  .aegis-hero .version-tag {
    display: inline-block;
    background: rgba(56,139,253,0.15);
    color: #388bfd;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 2px 10px;
    border-radius: 20px;
    border: 1px solid rgba(56,139,253,0.25);
    margin-bottom: 0.7rem;
    letter-spacing: 0.5px;
  }

  /* Risk badges */
  .aegis-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
  }
  .badge-low  { background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }
  .badge-med  { background: rgba(210,153,34,0.15); color: #e3b341; border: 1px solid rgba(210,153,34,0.3); }
  .badge-high { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid rgba(248,81,73,0.3); }

  /* Stat cards */
  .stat-card {
    background: rgba(22, 33, 54, 0.8);
    border: 1px solid rgba(56,139,253,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .stat-card .label {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.3rem;
  }
  .stat-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
  }
  .stat-card .value.green { color: #3fb950; }
  .stat-card .value.yellow { color: #e3b341; }
  .stat-card .value.red { color: #f85149; }
  .stat-card .value.blue { color: #58a6ff; }

  /* Section headers */
  .section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #e6edf3;
    border-left: 3px solid #388bfd;
    padding-left: 0.75rem;
    margin: 1.5rem 0 0.8rem 0;
  }

  /* Archaeology story cards */
  .story-card {
    background: rgba(56,139,253,0.06);
    border: 1px solid rgba(56,139,253,0.2);
    border-left: 4px solid #388bfd;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.6rem;
  }
  .story-card .headline {
    font-weight: 600;
    color: #e6edf3;
    font-size: 0.9rem;
  }
  .story-card .detail {
    color: #8b949e;
    font-size: 0.82rem;
    margin-top: 0.2rem;
  }

  /* Safety gate banner */
  .safety-banner {
    background: rgba(248,81,73,0.1);
    border: 1px solid rgba(248,81,73,0.35);
    border-radius: 10px;
    padding: 0.8rem 1.1rem;
    color: #f85149;
    font-weight: 500;
    font-size: 0.9rem;
    margin: 0.5rem 0;
  }

  /* Live metrics row */
  .live-metric {
    background: rgba(13, 17, 23, 0.8);
    border: 1px solid rgba(48,54,61,0.8);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    text-align: center;
  }
  .live-metric .lm-label { font-size: 0.7rem; color: #8b949e; letter-spacing: 0.5px; }
  .live-metric .lm-value { font-size: 1.15rem; font-weight: 600; color: #e6edf3; font-family: 'JetBrains Mono', monospace; }

  /* Factor tags */
  .factor-tag {
    display: inline-block;
    background: rgba(56,139,253,0.1);
    border: 1px solid rgba(56,139,253,0.2);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78rem;
    color: #8b949e;
    margin: 2px 3px 2px 0;
    font-family: 'JetBrains Mono', monospace;
  }

  div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
  div[data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)


def ensure_demo_environment(target: Path):
    if not target.exists():
        build_demo(target)
    total, used, _free = shutil.disk_usage(target)
    db.log_usage(str(target), used, total)
    if predictor.forecast(str(target), min_points=3) is None:
        predictor.seed_synthetic_history(str(target), total, current_used_bytes=used,
                                          daily_growth_gb=1.8, days_back=14)


if "bootstrapped" not in st.session_state:
    ensure_demo_environment(DEFAULT_TARGET)
    st.session_state.bootstrapped = True


@st.cache_resource(show_spinner=False)
def get_trained_model():
    X, y = ml_training.generate_training_data(samples=3000, seed=42)
    return future_usage_model.train_model(X, y)


def human(n_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def risk_badge_html(tier: str) -> str:
    cls = {"LOW": "badge-low", "MEDIUM": "badge-med", "HIGH": "badge-high"}.get(tier, "badge-med")
    return f'<span class="aegis-badge {cls}">{tier}</span>'


def make_pie(labels, values, colors, title=""):
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='rgba(0,0,0,0)', width=0)),
        textinfo="percent",
        textfont=dict(size=12, family="JetBrains Mono"),
        hovertemplate="<b>%{label}</b><br>%{value} files (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#8b949e"), x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3", family="Inter"),
        legend=dict(font=dict(size=11, color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=40, b=10, l=10, r=10),
        height=260,
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown("""
<div class="aegis-hero">
  <div class="version-tag">AEGISSTORE · RISK-ADAPTIVE AI · LINUX STORAGE</div>
  <h1>🛡️ AegisStore</h1>
  <p>AI understands what can be optimized. AegisStore decides whether it is safe to act — and learns from what you accept or reject.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
col_input, col_scan, col_reset = st.columns([3, 1, 1])
target_dir = col_input.text_input("Directory to scan", value="./demo_disk", label_visibility="collapsed")
col_input.caption("📁 Directory to scan")
scan_clicked = col_scan.button("🔍 Scan Now", use_container_width=True, type="primary")
reset_clicked = col_reset.button("♻️ Reset Demo", use_container_width=True,
                                  help="Wipes demo_disk, quarantine, and history, then rebuilds.")

with st.expander("🛡️ Safety Guarantees & Threat Model", expanded=False):
    st.markdown("""
    | Guarantee | How it works |
    |---|---|
    | 🗑️ No direct deletion | All cleanup goes through controlled quarantine |
    | 🔒 Open file protection | Active process files are never touched |
    | ⚠️ Risk-threshold gating | Every file scored 0–100 before any action |
    | 🖥️ Live-load deferral | High CPU/RAM/IO defers cleanup in real time |
    | 📦 Dependency awareness | Package-owned, Git-tracked, symlinks all flagged |
    | 🎯 Feedback calibration | Accept/reject clicks retrain risk boundaries |
    """)

if reset_clicked:
    target = Path(target_dir)
    if target.exists():
        shutil.rmtree(target)
    for p in [Path("quarantine"), Path("aegisstore.db"), Path("calibration.json")]:
        if p.exists():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ---------------------------------------------------------------------------
# Live System Load
# ---------------------------------------------------------------------------
load = safety_gate.read_system_load(sample_seconds=0.3)
busy = safety_gate.is_system_busy(load)
state = str(load.get("state", "NORMAL"))
state_color = {"NORMAL": "#3fb950", "BUSY": "#e3b341", "CRITICAL": "#f85149"}.get(state, "#8b949e")

st.markdown('<div class="section-header">Live System Monitor</div>', unsafe_allow_html=True)
lc1, lc2, lc3, lc4, lc5 = st.columns(5)
for col, label, val, unit in [
    (lc1, "CPU", f"{load['cpu_percent']:.0f}", "%"),
    (lc2, "RAM", f"{load['memory_percent']:.0f}", "%"),
    (lc3, "Disk Read", f"{load['disk_read_mb_s']:.1f}", " MB/s"),
    (lc4, "Disk Write", f"{load['disk_write_mb_s']:.1f}", " MB/s"),
]:
    col.markdown(f"""
    <div class="live-metric">
      <div class="lm-label">{label}</div>
      <div class="lm-value">{val}<span style="font-size:0.7rem;color:#8b949e">{unit}</span></div>
    </div>""", unsafe_allow_html=True)

lc5.markdown(f"""
<div class="live-metric">
  <div class="lm-label">Safety Gate</div>
  <div class="lm-value" style="color:{state_color}">{state}</div>
</div>""", unsafe_allow_html=True)
st.caption(f"I/O Wait: {load['io_wait_percent']:.0f}%  ·  Thresholds — CPU busy ≥75% critical ≥90% | RAM busy ≥80% critical ≥90% | IO busy ≥10% critical ≥20%")

if busy:
    st.markdown(f"""<div class="safety-banner">
    ⚡ LIVE SAFETY OVERRIDE ACTIVE — automatic cleanup deferred.
    CPU {load['cpu_percent']:.0f}% · RAM {load['memory_percent']:.0f}% · IO Wait {load['io_wait_percent']:.0f}%
    </div>""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.summary = None
    st.session_state.reclaimable = 0

# ---------------------------------------------------------------------------
# Scan Pipeline
# ---------------------------------------------------------------------------
if scan_clicked:
    target = str(Path(target_dir))
    if not Path(target).exists():
        st.error(f"'{target}' does not exist. Run `python3 demo_setup.py {target}` first.")
    else:
        progress = st.progress(0)
        status = st.empty()

        status.text("📁 Scanning filesystem...")
        total, used, _free = shutil.disk_usage(target)
        db.log_usage(target, used, total)
        records = scanner.scan_and_classify(target)
        progress.progress(20)

        status.text("📊 Analyzing usage intelligence...")
        analyzed = usage_intelligence.analyze_records(records)
        progress.progress(35)

        status.text("🤖 Predicting future usage (ML)...")
        model = get_trained_model()
        predicted = [future_usage_model.predict_record(model, r) for r in analyzed]
        progress.progress(55)

        status.text("🧠 Generating recommendations...")
        candidates, reclaimable = scanner.reclaimable_summary(records)
        candidate_paths = {str(c["path"]) for c in candidates}
        candidate_records = [r for r in predicted if str(r["path"]) in candidate_paths]
        candidate_records = sorted(candidate_records, key=lambda r: r["size_bytes"], reverse=True)[:20]
        progress.progress(70)

        status.text("⚙️ Computing risk-adaptive decisions...")
        rows = []
        for r in candidate_records:
            ctx = context.enrich(str(r["path"]))
            merged = {**r, **ctx}
            decision = decision_engine.assess(r, ctx, load, busy)
            rec = recommendation_engine.recommend(merged)

            if decision["action"] == "DEFER":
                db.log_schedule_event(r["path"], "DEFERRED", load, reason=decision["reason"])
            else:
                previous = [a for a in db.recent_audit(limit=50)
                            if a["path"] == str(r["path"]) and a["action"] == "DEFERRED"]
                if previous:
                    db.log_schedule_event(r["path"], "RETRIED", load,
                                           reason="System load is now within safe limits.")

            cid = db.save_candidate(r)
            db.save_decision(cid, {**ctx, "cpu_percent": load["cpu_percent"],
                                    "io_wait_percent": load["io_wait_percent"], **decision})

            rows.append({
                "File": r["path"].name,
                "Path": str(r["path"]),
                "Size": human(r["size_bytes"]),
                "size_bytes": r["size_bytes"],
                "Age (days)": r["age_days"],
                "Classification": r["classification"],
                "Confidence": f"{r['confidence']:.0%}",
                "Active": ctx["active_process"],
                "Pkg-owned": ctx["package_owned"],
                "Git-tracked": ctx["git_tracked"],
                "risk_score": decision["risk_score"],
                "risk_tier": decision["risk_tier"],
                "action": decision["action"],
                "reason": decision["reason"],
                "factors": decision.get("factors", []),
                "Risk": decision["risk_tier"],
                "Risk Score": f"{decision['risk_score']} / 100",
                "Action": decision["action"],
                "Reason": decision["reason"],
                "usage_profile": r.get("usage_profile"),
                "future_usage_probability": r.get("future_usage_probability"),
                "future_usage_class": r.get("future_usage_class"),
                "future_usage_explanation": r.get("future_usage_explanation"),
                "recommendation": rec.get("recommendation"),
                "recommendation_reason": rec.get("recommendation_reason"),
                "safety_flags": rec.get("safety_flags", []),
                "safety_blocked": rec.get("safety_blocked", False),
            })

        progress.progress(88)
        status.text("📖 Generating forecasts & stories...")

        st.session_state.results = rows
        st.session_state.archaeology_records = analyzed
        st.session_state.reclaimable = reclaimable
        st.session_state.total_disk = total
        st.session_state.used_disk = used
        st.session_state.target = target

        automated = [r for r in rows if r["Action"] == "AUTOMATE"]
        deferred = [r for r in rows if r["Action"] == "DEFER"]
        avg_conf = (sum(float(r["Confidence"].strip("%")) for r in rows) / 100 / len(rows)) if rows else 0
        fc = predictor.forecast(target)
        fc_detailed = storage_intelligence.storage_forecast_detailed(target, reclaimable_bytes=reclaimable)

        summary = {
            "total_candidates": len(records),
            "total_reclaimable_gb": reclaimable / (1024 ** 3),
            "top_reason": "cold/redundant data",
            "deferred_count": len(deferred),
            "automated_count": len(automated),
            "avg_confidence": avg_conf,
        }
        if fc:
            summary["growth_rate_gb_per_day"] = fc["growth_rate_gb_per_day"]
            summary["days_to_90pct"] = fc["predictions_days"].get(0.90)
            st.session_state.forecast = fc
        else:
            st.session_state.forecast = None
        st.session_state.forecast_detailed = fc_detailed
        st.session_state.summary = summary

        progress.progress(100)
        status.text("✅ Scan complete!")
        st.success(f"✅ Analyzed {len(records)} files — {len(rows)} candidates — {len(automated)} ready for cleanup.")
        time.sleep(0.4)
        progress.empty()
        status.empty()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if st.session_state.results is not None:
    used = st.session_state.used_disk
    total = st.session_state.total_disk
    results = st.session_state.results

    # Top metrics
    st.markdown('<div class="section-header">Scan Summary</div>', unsafe_allow_html=True)
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Disk Usage", f"{used/total:.0%}", help=f"{human(used)} / {human(total)}")
    sm2.metric("Reclaimable", human(st.session_state.reclaimable))
    sm3.metric("Candidates", len(results))
    sm4.metric("Ready to Clean", sum(1 for r in results if r["Action"] == "AUTOMATE"))

    # ---- PIE CHARTS ROW ----
    st.markdown('<div class="section-header">Visual Breakdown</div>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)

    # Pie 1: Risk Tier
    risk_counts = {t: sum(1 for r in results if r.get("risk_tier") == t) for t in ["LOW", "MEDIUM", "HIGH"]}
    with pc1:
        if any(risk_counts.values()):
            fig1 = make_pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                colors=["#3fb950", "#e3b341", "#f85149"],
                title="Risk Tier Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    # Pie 2: File Classification
    clf_counts = {}
    for r in results:
        c = r.get("Classification", "Unknown")
        clf_counts[c] = clf_counts.get(c, 0) + 1
    clf_colors = {"HOT": "#f85149", "WARM": "#e3b341", "COLD": "#388bfd", "REDUNDANT": "#8b949e", "Unknown": "#30363d"}
    with pc2:
        if clf_counts:
            fig2 = make_pie(
                labels=list(clf_counts.keys()),
                values=list(clf_counts.values()),
                colors=[clf_colors.get(k, "#58a6ff") for k in clf_counts.keys()],
                title="File Classification"
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Pie 3: Recommended Action
    rec_counts = {}
    for r in results:
        a = r.get("recommendation") or r.get("Action", "UNKNOWN")
        rec_counts[a] = rec_counts.get(a, 0) + 1
    rec_colors_map = {"CLEANUP": "#f85149", "ARCHIVE": "#e3b341", "KEEP": "#3fb950",
                      "REVIEW": "#388bfd", "AUTOMATE": "#3fb950", "DEFER": "#8b949e",
                      "SKIP": "#30363d", "APPROVAL_REQUIRED": "#e3b341"}
    with pc3:
        if rec_counts:
            fig3 = make_pie(
                labels=list(rec_counts.keys()),
                values=list(rec_counts.values()),
                colors=[rec_colors_map.get(k, "#58a6ff") for k in rec_counts.keys()],
                title="Recommended Actions"
            )
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # Risk threshold caption
    thresholds = decision_engine.current_thresholds()
    cal_note = "default" if thresholds["is_default"] else "recalibrated from feedback"
    st.caption(f"Risk boundary: LOW < {thresholds['low_threshold']} · HIGH ≥ {thresholds['high_threshold']} ({cal_note})")

    if busy:
        st.markdown(f"""<div class="safety-banner">
        ⚡ LIVE SAFETY OVERRIDE ACTIVE — cleanup deferred.
        CPU {load['cpu_percent']:.0f}% · RAM {load['memory_percent']:.0f}% · IO {load['io_wait_percent']:.0f}%
        </div>""", unsafe_allow_html=True)

    # ---- GROWTH FORECAST ----
    fc_detail = st.session_state.get("forecast_detailed")
    if fc_detail:
        fc = fc_detail["forecast"]
        st.markdown('<div class="section-header">📈 Growth Forecast</div>', unsafe_allow_html=True)
        if fc:
            fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)
            fcol1.metric("Current Usage", f"{fc['current_usage_pct']*100:.0f}%")
            fcol2.metric("Growth Rate", fc_detail["growth_rate_formatted"])
            d85 = fc["predictions_days"].get(0.85)
            fcol3.metric("Days to 85%", f"{d85:.0f}" if d85 else "N/A")
            d90 = fc["predictions_days"].get(0.90)
            fcol4.metric("Days to 90%", f"{d90:.0f}" if d90 else "N/A")
            d95 = fc["predictions_days"].get(0.95)
            fcol5.metric("Days to 95%", f"{d95:.0f}" if d95 else "N/A")
            st.metric("Forecast Quality", fc_detail["forecast_quality"])

            history_rows = db.usage_series(st.session_state.target)
            if len(history_rows) >= 2:
                chart_df = pd.DataFrame([{
                    "Date": datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d"),
                    "Historical Usage (GB)": r["used_bytes"] / (1024 ** 3),
                } for r in history_rows])
                chart_df = chart_df.drop_duplicates(subset="Date", keep="last").set_index("Date")
                st.line_chart(chart_df, height=200)

            if fc_detail["cleanup_impact"]:
                st.markdown('<div class="section-header">💾 Cleanup Impact Projection</div>', unsafe_allow_html=True)
                ci = fc_detail["cleanup_impact"]
                ci1, ci2, ci3, ci4 = st.columns(4)
                ci1.metric("Current Usage", ci["current_used_formatted"])
                ci2.metric("Reclaimable", ci["reclaimable_formatted"])
                ci3.metric("After Cleanup", ci["estimated_after_formatted"])
                ci4.metric("After Cleanup %", f"{ci['estimated_after_pct']*100:.0f}%")
                st.caption("Estimated impact if all reclaimable data were safely removed.")

        st.markdown('<div class="section-header">🧭 Storage Intelligence</div>', unsafe_allow_html=True)
        st.info(fc_detail["recommendation"])

    # ---- DIGITAL ARCHAEOLOGY ----
    st.markdown('<div class="section-header">📜 Digital Archaeology</div>', unsafe_allow_html=True)
    st.caption("Your storage, told as a story a non-technical reader can act on in one glance.")
    archaeology_records = st.session_state.get("archaeology_records", [])
    stories = archaeology.build_stories(archaeology_records) if archaeology_records else []
    if stories:
        for s in stories[:6]:
            st.markdown(f"""
            <div class="story-card">
              <div class="headline">{s['headline']}</div>
              <div class="detail">{s['detail']}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No large enough groupings to narrate yet.")

# ---------------------------------------------------------------------------
# Candidate Table
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header">📋 Candidate Results — Risk-Adaptive Decisions</div>', unsafe_allow_html=True)
results = st.session_state.get("results") or []
df = pd.DataFrame(results)

if not df.empty:
    display_df = pd.DataFrame([{
        "File": r["File"],
        "Size": r["Size"],
        "Age (days)": r["Age (days)"],
        "Classification": r["Classification"],
        "Usage Profile": r.get("usage_profile") or "—",
        "Future Use %": (f"{r['future_usage_probability']*100:.1f}%"
                         if r.get("future_usage_probability") is not None else "—"),
        "Risk Score": r["Risk Score"],
        "Risk": r["Risk"],
        "Recommendation": r.get("recommendation") or r["Action"],
        "Reason": r.get("recommendation_reason") or r["Reason"],
    } for r in results])

    def risk_color(val):
        return {
            "LOW": "background-color:#1a3a1f; color:#3fb950",
            "MEDIUM": "background-color:#3a2e0a; color:#e3b341",
            "HIGH": "background-color:#3a0f0f; color:#f85149",
        }.get(val, "")

    st.dataframe(display_df.style.map(risk_color, subset=["Risk"]), use_container_width=True, hide_index=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Report (CSV)", data=csv_bytes,
                        file_name=f"aegisstore_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv")

    # ---- WHY THIS DECISION ----
    st.markdown('<div class="section-header">🔍 Why This Decision?</div>', unsafe_allow_html=True)
    candidate_names = [r["File"] for r in results]
    selected_file = st.selectbox("Select a candidate to inspect", candidate_names, index=0)
    selected = next((r for r in results if r["File"] == selected_file), None)

    if selected:
        badge = risk_badge_html(selected["Risk"])
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Risk Score", selected["Risk Score"])
        col_b.metric("Action", selected["Action"])
        col_c.metric("Age", f"{selected['Age (days)']} days")

        st.markdown(f"**File:** `{selected['File']}`  {badge}", unsafe_allow_html=True)

        if selected.get("recommendation"):
            st.markdown(f"**ML Recommendation:** `{selected['recommendation']}` — {selected.get('recommendation_reason', '')}")
        if selected.get("future_usage_probability") is not None:
            st.markdown(
                f"**Predicted Future Use:** `{selected['future_usage_probability']*100:.1f}%` "
                f"({selected.get('future_usage_class', '—')}) — {selected.get('future_usage_explanation', '')}"
            )

        st.markdown("**Scoring Factors:**")
        factors = selected.get("factors", [])
        if factors:
            tags_html = "".join(f'<span class="factor-tag">{f}</span>' for f in factors)
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.caption("No explanatory factors available.")
        st.markdown(f"**Decision:** {selected['Reason']}")

        st.divider()
        st.markdown("#### 🔄 Counterfactual — What Would Change This Decision?")
        try:
            from aegisstore import counterfactual
            original = next((c for c in scanner.scan_and_classify(st.session_state.target)
                              if str(c["path"]) == selected["Path"]), None)
            if original:
                cf = counterfactual.explain_age_change(
                    original, context.enrich(str(original["path"])), load, busy, days_delta=-7)
                cfc1, cfc2 = st.columns(2)
                cfc1.metric("Current Risk", f"{cf['current_score']} / 100")
                cfc2.metric("If 7 Days Newer", f"{cf['counterfactual_score']} / 100", delta=cf["delta"])
                st.info(f"💡 {cf['explanation']}")
                if cf["current_action"] != cf["counterfactual_action"]:
                    st.warning(f"Decision would change: **{cf['current_action']}** → **{cf['counterfactual_action']}**")
                else:
                    st.caption(f"Decision remains: {cf['current_action']}")
        except Exception as e:
            st.warning(f"Counterfactual unavailable: {e}")

        st.divider()
        st.markdown("#### 🎓 Was this recommendation right?")
        st.caption("Your feedback retrains the risk-tier boundary via logistic regression.")
        fb1, fb2, fb3 = st.columns([1, 1, 3])
        rec_label = selected.get("recommendation") or selected["Action"]
        if fb1.button("👍 Accept", key=f"accept_{selected['Path']}"):
            db.log_recommendation_feedback(selected["Path"], rec_label, selected["risk_score"],
                                            selected.get("future_usage_probability"), accepted=True)
            st.success("Feedback logged: accepted.")
        if fb2.button("👎 Reject", key=f"reject_{selected['Path']}"):
            db.log_recommendation_feedback(selected["Path"], rec_label, selected["risk_score"],
                                            selected.get("future_usage_probability"), accepted=False)
            st.success("Feedback logged: rejected.")
        fb3.caption(f"{db.recommendation_feedback_count()} feedback samples logged (need 6+ to recalibrate).")

    st.markdown("#### 🧠 Recalibrate Risk Thresholds")
    st.caption("Refits LOW/MEDIUM/HIGH boundaries from your accept/reject clicks — real logistic regression, not a reset.")
    rc1, rc2 = st.columns([1, 3])
    if rc1.button("Recalibrate Now"):
        import recalibrate as recalibrate_module
        result = recalibrate_module.recalibrate_from_feedback()
        if result["status"] == "recalibrated":
            st.success(
                f"Recalibrated from {result['sample_count']} samples. "
                f"LOW: {result['old_thresholds']['low_threshold']} → {result['new_thresholds']['low_threshold']} · "
                f"HIGH: {result['old_thresholds']['high_threshold']} → {result['new_thresholds']['high_threshold']}"
            )
            st.caption("Re-run a scan to see new boundaries applied.")
        elif result["status"] == "insufficient_data":
            st.info(f"Only {result['sample_count']} samples — need at least 6.")
        elif result["status"] == "no_variation":
            st.info("Need at least one accept and one reject before recalibrating.")
        else:
            st.warning(f"Recalibration did not run: {result.get('error', result['status'])}")
    if rc2.button("Reset to Defaults"):
        import recalibrate as recalibrate_module
        recalibrate_module.reset_calibration()
        st.success("Calibration reset to defaults (LOW < 31, HIGH ≥ 66).")

else:
    st.info("Run a scan to see candidate results.")

# ---------------------------------------------------------------------------
# Take Action
# ---------------------------------------------------------------------------
if results:
    st.markdown('<div class="section-header">⚙️ Take Action</div>', unsafe_allow_html=True)
    auto_eligible = [r for r in results if r["Action"] == "AUTOMATE"]
    scheduled = [r for r in results if r["Action"] == "SCHEDULE"]
    approval = [r for r in results if r["Action"] == "APPROVAL_REQUIRED"]
    deferred = [r for r in results if r["Action"] == "DEFER"]
    skipped = [r for r in results if r["Action"] == "SKIP"]

    if auto_eligible:
        st.info(f"🟢 {len(auto_eligible)} candidates are safe for automatic cleanup.")
        col_batch, col_safety = st.columns([2, 1])
        if col_batch.button("⚡ Execute Batch Cleanup", type="primary",
                             help="Quarantine all AUTOMATE-eligible files with SHA-256 integrity verification"):
            batch_candidates = [{"path": r["Path"], "reason": r["Reason"]} for r in auto_eligible]
            batch_result = executor.batch_quarantine(batch_candidates, load, verify_safety=True)
            if batch_result["safety_cleared"]:
                if batch_result["executed"]:
                    st.success(
                        f"✓ {len(batch_result['executed'])} files quarantined · "
                        f"Recovered {batch_result['total_bytes_recovered'] / (1024**3):.2f} GB"
                    )
                    st.balloons()
                if batch_result["failed"]:
                    st.warning(f"⚠ {len(batch_result['failed'])} files could not be quarantined.")
            else:
                st.warning(
                    f"🛑 Safety gate blocked batch cleanup — system busy "
                    f"(CPU {load['cpu_percent']:.0f}%, RAM {load['memory_percent']:.0f}%). "
                    f"All {len(batch_result['skipped'])} candidates remain safe."
                )
        col_safety.metric("Batch Safety", "✓ PASS" if not busy else "⚠ DEFER")

    for action_name, items, guidance in [
        ("AUTOMATE", auto_eligible, "Ready for automatic quarantine."),
        ("SCHEDULE", scheduled, "Handle during a safer window."),
        ("APPROVAL_REQUIRED", approval, "Requires explicit user approval."),
        ("DEFER", deferred, "System busy — do not execute now."),
        ("SKIP", skipped, "Safety checks blocked — no action."),
    ]:
        if items:
            badge_html = risk_badge_html({"AUTOMATE": "LOW", "SCHEDULE": "MEDIUM",
                                          "APPROVAL_REQUIRED": "MEDIUM", "DEFER": "HIGH", "SKIP": "HIGH"}.get(action_name, "MEDIUM"))
            st.markdown(f"**{action_name}** {badge_html} — {guidance}", unsafe_allow_html=True)
            for r in items:
                c1, c2 = st.columns([4, 1])
                c1.write(f"`{r['File']}` · {r['Size']} · {r['Reason']}")
                if action_name == "AUTOMATE":
                    if c2.button("Quarantine", key=f"q_{r['Path']}"):
                        try:
                            info = executor.quarantine_file(r["Path"], r["Reason"])
                            st.success(f"Quarantined · integrity_verified={info['integrity_verified']}")
                        except FileNotFoundError:
                            st.warning("Already quarantined or moved.")
                else:
                    c2.write("—")

    st.markdown('<div class="section-header">📖 Storage Story</div>', unsafe_allow_html=True)
    with st.spinner("Generating narrative..."):
        story = storage_story.generate_story(st.session_state.summary)
    st.info(story)

# ---------------------------------------------------------------------------
# Recovery & Quarantine Management
# ---------------------------------------------------------------------------
st.divider()
st.markdown('<div class="section-header">🧾 Recovery & Quarantine Management</div>', unsafe_allow_html=True)
recovery_stats = executor.recovery_stats()
q1, q2, q3 = st.columns(3)
q1.metric("Files in Quarantine", recovery_stats["file_count"])
q2.metric("Total Size", f"{recovery_stats['total_bytes'] / (1024**3):.2f} GB")
q3.metric("Integrity OK", f"{recovery_stats['integrity_ok']}/{recovery_stats['file_count']}")

quarantine_items = executor.list_quarantine(limit=50)
if quarantine_items:
    quar_df = pd.DataFrame([{
        "File": Path(item["original_path"]).name,
        "Original Path": item["original_path"],
        "Reason": item["reason"],
        "Size (GB)": f"{item['size_bytes'] / (1024**3):.2f}",
        "Integrity": "✓" if item["integrity_verified"] else "✗",
    } for item in quarantine_items])
    st.dataframe(quar_df, use_container_width=True, hide_index=True)

    selected_quar = st.selectbox("Select a quarantined file to recover",
                                  options=[item["quarantine_path"] for item in quarantine_items],
                                  format_func=lambda p: Path(p).name)
    if selected_quar and st.button("↩️ Restore to Original Location"):
        try:
            restore_result = executor.undo_last(selected_quar)
            st.success(f"Restored: {restore_result['restored_to']}")
            st.rerun()
        except Exception as e:
            st.error(f"Recovery failed: {e}")
else:
    st.info("No files in quarantine.")

# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header">📜 Audit Log</div>', unsafe_allow_html=True)
audit_rows = db.recent_audit(limit=15)
if audit_rows:
    audit_df = pd.DataFrame([dict(r) for r in audit_rows])
    st.dataframe(audit_df[["event_time", "action", "path", "reversible", "detail"]],
                 use_container_width=True, hide_index=True)
else:
    st.caption("No actions taken yet.")

# ---------------------------------------------------------------------------
# Scheduling Timeline
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header">🕐 Performance-Aware Scheduling Timeline</div>', unsafe_allow_html=True)
schedule_actions = {"DEFERRED", "RETRIED", "EXECUTED", "QUARANTINE"}
schedule_rows = [dict(r) for r in db.recent_audit(limit=50) if r["action"] in schedule_actions]
if schedule_rows:
    timeline = [{
        "Time": datetime.fromtimestamp(r["event_time"]).strftime("%H:%M:%S"),
        "Event": "EXECUTED" if r["action"] == "QUARANTINE" else r["action"],
        "File": r["path"],
        "Reason": r["detail"],
    } for r in schedule_rows]
    st.dataframe(pd.DataFrame(timeline), use_container_width=True, hide_index=True)
else:
    st.info("No scheduling events yet. Run a scan and try quarantining a file.")
