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

st.set_page_config(page_title="AegisStore Panel", page_icon="🛡️", layout="wide")
db.init_db()

DEFAULT_TARGET = Path("./demo_disk")

# ---------------------------------------------------------------------------
# CSS — Ultra-Clean, Production-Ready Systems Workspace Aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&family=Geist:wght@300;400;500;600;700&display=swap');

  /* Global structural variables */
  :root {
    --bg-primary: #090a0f;
    --bg-surface: #12131a;
    --border-subtle: #222530;
    --text-main: #f4f5f6;
    --text-muted: #888e96;
    --accent-blue: #2f80ed;
  }

  html, body, [class*="css"], div[data-testid="stAppViewContainer"] { 
    font-family: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif; 
    background-color: var(--bg-primary) !important;
    color: var(--text-main);
  }

  /* Structural Containers */
  div[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle);
  }

  /* Refined Header Area */
  .aegis-header-container {
    padding: 1.5rem 0 2rem 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 2rem;
  }
  .aegis-header-container h1 {
    font-weight: 700;
    font-size: 2.2rem;
    letter-spacing: -0.03em;
    color: var(--text-main);
    margin: 0;
  }
  .aegis-header-container p {
    color: var(--text-muted);
    font-size: 1rem;
    margin-top: 0.4rem;
    max-width: 750px;
    line-height: 1.5;
  }
  .system-meta-badge {
    display: inline-flex;
    align-items: center;
    background: #171923;
    color: #4ba3e3;
    font-size: 0.75rem;
    font-family: 'Geist Mono', monospace;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 4px;
    border: 1px solid #232d42;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* Unified Workspace Card Grid Component */
  .workspace-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
  }

  /* Minimal High-Contrast Metric Block */
  .metric-block {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
  }
  .metric-block .metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
  }
  .metric-block .metric-value {
    font-size: 1.75rem;
    font-weight: 600;
    color: var(--text-main);
    font-family: 'Geist Mono', monospace;
    letter-spacing: -0.02em;
  }
  
  /* Precision Semantic Indicators */
  .color-indicator-green { color: #10b981 !important; }
  .color-indicator-amber { color: #f59e0b !important; }
  .color-indicator-red { color: #ef4444 !important; }
  .color-indicator-blue { color: #3b82f6 !important; }

  /* Section Title Elements */
  .section-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin: 2.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-subtle);
  }

  /* Refined Risk Badges */
  .aegis-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'Geist Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.02em;
  }
  .badge-low  { background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.2); }
  .badge-med  { background: rgba(245, 158, 11, 0.1); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.2); }
  .badge-high { background: rgba(239, 68, 68, 0.1); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.2); }

  /* Narrative Archaeology Blocks */
  .narrative-block {
    background: #0f111a;
    border-left: 2px solid var(--border-subtle);
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
  }
  .narrative-block .title {
    font-weight: 500;
    color: var(--text-main);
    font-size: 0.9rem;
  }
  .narrative-block .desc {
    color: var(--text-muted);
    font-size: 0.8rem;
    margin-top: 0.25rem;
    line-height: 1.4;
  }

  /* Critical Safety Overrides Banner */
  .safety-override-banner {
    background: rgba(239, 68, 68, 0.06);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 6px;
    padding: 1rem;
    color: #f87171;
    font-weight: 400;
    font-size: 0.88rem;
    margin: 1.5rem 0;
    line-height: 1.5;
  }

  /* Utility Micro-Tags */
  .micro-tag {
    display: inline-block;
    background: #1c1e29;
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 0.72rem;
    color: #a3a8b4;
    margin-right: 4px;
    font-family: 'Geist Mono', monospace;
  }

  /* Streamlit native widget customization to blend cleanly */
  div[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-family: 'Geist Mono', monospace !important; font-weight:600 !important; }
  div[data-testid="stMetricLabel"] { font-size: 0.75rem !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
  .stDataFrame, div[data-testid="stTable"] { border: 1px solid var(--border-subtle) !important; border-radius: 6px !important; overflow: hidden; }
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
        hole=0.7,
        marker=dict(
            colors=colors,
            line=dict(color='#12131a', width=2)
        ),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value} files (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title.upper(), font=dict(size=11, color="#888e96", family="Geist", weight=600), x=0.0, y=0.98),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f4f5f6", family="Geist"),
        legend=dict(
            font=dict(size=11, color="#888e96", family="Geist Mono"),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            x=0, y=-0.1
        ),
        margin=dict(t=40, b=40, l=0, r=0),
        height=220,
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# PLATFORM CLEAN HEADER AREA
# ---------------------------------------------------------------------------
st.markdown("""
<div class="aegis-header-container">
  <div class="system-meta-badge">Core Engine Engine · Risk Adaptive Architecture · v2.4.1</div>
  <h1>🛡️ AegisStore Terminal</h1>
  <p>An automated context-driven runtime safety layer for cloud storage environments. The panel analyzes historical telemetry patterns to build proactive counterfactual cleanup schedules while guaranteeing real-time infrastructure runtime consistency.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Workspace Control Row
# ---------------------------------------------------------------------------
col_input, col_scan, col_reset = st.columns([3, 1, 1])
target_dir = col_input.text_input("Directory to scan", value="./demo_disk", label_visibility="collapsed")
col_input.caption("Target Environment Directory Mount Path")
scan_clicked = col_scan.button("Run Telemetry Scan", use_container_width=True, type="primary")
reset_clicked = col_reset.button("Reset Storage State", use_container_width=True,
                                  help="Wipes demo_disk, quarantine, and database history states.")

with st.expander("Telemetry Structural Safety Context Definitions", expanded=False):
    st.markdown("""

    | Operational Guarantee | Structural Mechanism |
    | :--- | :--- |
    | **Quarantine Isolation Topology** | Deletion handles are deferred; targets are shifted to verification staging partitions. |
    | **Runtime File Lock Interlocking** | File handles mapped to running standard active PIDs are instantly dropped from scopes. |
    | **Boundary Threshold Constraints** | Strict individual element evaluation scores bounded between 0–100 before handling triggers. |
    | **Load-Factor Throttling Enforcer** | High real-time CPU/RAM footprint instantly defers queue items back to schedule queues. |
    | **Dependency Trace Analysis** | Validates package managers, workspace configurations, and submodules before cataloging. |
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
# System Live Metrics Layout Configuration
# ---------------------------------------------------------------------------
load = safety_gate.read_system_load(sample_seconds=0.3)
busy = safety_gate.is_system_busy(load)
state = str(load.get("state", "NORMAL"))
state_color_class = {"NORMAL": "color-indicator-green", "BUSY": "color-indicator-amber", "CRITICAL": "color-indicator-red"}.get(state, "")

st.markdown('<div class="section-title">Telemetry Environment Monitor</div>', unsafe_allow_html=True)
lc1, lc2, lc3, lc4, lc5 = st.columns(5)

with lc1:
    st.markdown(f'<div class="workspace-card"><div class="metric-block"><div class="metric-label">CPU Footprint</div><div class="metric-value">{load["cpu_percent"]:.0f}<span style="font-size:0.8rem;color:var(--text-muted)">%</span></div></div></div>', unsafe_allow_html=True)
with lc2:
    st.markdown(f'<div class="workspace-card"><div class="metric-block"><div class="metric-label">RAM Saturation</div><div class="metric-value">{load["memory_percent"]:.0f}<span style="font-size:0.8rem;color:var(--text-muted)">%</span></div></div></div>', unsafe_allow_html=True)
with lc3:
    st.markdown(f'<div class="workspace-card"><div class="metric-block"><div class="metric-label">I/O Bandwidth Read</div><div class="metric-value">{load["disk_read_mb_s"]:.1f}<span style="font-size:0.8rem;color:var(--text-muted)"> MB/s</span></div></div></div>', unsafe_allow_html=True)
with lc4:
    st.markdown(f'<div class="workspace-card"><div class="metric-block"><div class="metric-label">I/O Bandwidth Write</div><div class="metric-value">{load["disk_write_mb_s"]:.1f}<span style="font-size:0.8rem;color:var(--text-muted)"> MB/s</span></div></div></div>', unsafe_allow_html=True)
with lc5:
    st.markdown(f'<div class="workspace-card"><div class="metric-block"><div class="metric-label">Safety Status</div><div class="metric-value {state_color_class}">{state}</div></div></div>', unsafe_allow_html=True)

st.caption(f"Kernel I/O Wait Ratio: {load['io_wait_percent']:.0f}%  ·  System Bounds Trigger Logic: CPU Limit [75% | 90%] — RAM Limit [80% | 90%] — I/O Wait Bound [10% | 20%]")

if busy:
    st.markdown(f"""<div class="safety-override-banner">
    <strong>⚠️ SYSTEM SAFETY REGULATOR OVERRIDE ACTIVE</strong><br/>
    Automated execution stacks are currently deferred into holding sequences. 
    Current metrics: Host Core Utilization at {load['cpu_percent']:.0f}%, Volatile RAM usage at {load['memory_percent']:.0f}%, System I/O wait times sit at {load['io_wait_percent']:.0f}%.
    </div>""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.summary = None
    st.session_state.reclaimable = 0

# ---------------------------------------------------------------------------
# Pipeline Engine Processing Flow Execution
# ---------------------------------------------------------------------------
if scan_clicked:
    target = str(Path(target_dir))
    if not Path(target).exists():
        st.error(f"Target file system scope path context '{target}' cannot be parsed by kernel mount layers.")
    else:
        progress = st.progress(0)
        status = st.empty()

        status.text("Mapping targeted workspace tree structures...")
        total, used, _free = shutil.disk_usage(target)
        db.log_usage(target, used, total)
        records = scanner.scan_and_classify(target)
        progress.progress(20)

        status.text("Evaluating deep file classification descriptors...")
        analyzed = usage_intelligence.analyze_records(records)
        progress.progress(35)

        status.text("Evaluating statistical models on access probability maps...")
        model = get_trained_model()
        predicted = [future_usage_model.predict_record(model, r) for r in analyzed]
        progress.progress(55)

        status.text("Filtering metadata structural target collections...")
        candidates, reclaimable = scanner.reclaimable_summary(records)
        candidate_paths = {str(c["path"]) for c in candidates}
        candidate_records = [r for r in predicted if str(r["path"]) in candidate_paths]
        candidate_records = sorted(candidate_records, key=lambda r: r["size_bytes"], reverse=True)[:20]
        progress.progress(70)

        status.text("Running adaptive multidimensional risk scoring tables...")
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
        status.text("Compiling historical trends projections...")

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
        status.empty()
        st.toast(f"Parsed {len(records)} node records cleanly.")
        time.sleep(0.2)
        progress.empty()

# ---------------------------------------------------------------------------
# Telemetry Output Layout Formatting Structures
# ---------------------------------------------------------------------------
if st.session_state.results is not None:
    used = st.session_state.used_disk
    total = st.session_state.total_disk
    results = st.session_state.results

    # Execution Top Summary Bar
    st.markdown('<div class="section-title">Analysis Index Vectors</div>', unsafe_allow_html=True)
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Storage Volume Allocation", f"{used/total:.1%}", delta=f"{human(used)} / {human(total)} used", delta_color="off")
    sm2.metric("Purgeable Volume Weight", human(st.session_state.reclaimable))
    sm3.metric("Evaluated Target Nodes", len(results))
    sm4.metric("Unconditional Clearance Queue", sum(1 for r in results if r["Action"] == "AUTOMATE"))

    # Analytical Distributive Plots Configuration Row
    st.markdown('<div class="section-title">Distribution Mappings</div>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)

    risk_counts = {t: sum(1 for r in results if r.get("risk_tier") == t) for t in ["LOW", "MEDIUM", "HIGH"]}
    with pc1:
        if any(risk_counts.values()):
            fig1 = make_pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                colors=["#10b981", "#f59e0b", "#ef4444"], 
                title="Risk Target Metrics Classification"
            )
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    clf_counts = {}
    for r in results:
        c = r.get("Classification", "Unknown")
        clf_counts[c] = clf_counts.get(c, 0) + 1
    clf_colors = {
        "HOT": "#ef4444", "WARM": "#f59e0b", "COLD": "#3b82f6",
        "REDUNDANT": "#8b5cf6", "Unknown": "#4b5563", "Cold + Redundant": "#14b8a6",
    }
    with pc2:
        if clf_counts:
            fig2 = make_pie(
                labels=list(clf_counts.keys()),
                values=list(clf_counts.values()),
                colors=[clf_colors.get(k, "#4b5563") for k in clf_counts.keys()],
                title="Data Lifecycle Profiles Index"
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    rec_counts = {}
    for r in results:
        a = r.get("recommendation") or r.get("Action", "UNKNOWN")
        rec_counts[a] = rec_counts.get(a, 0) + 1
    rec_colors_map = {
        "CLEANUP": "#ef4444", "ARCHIVE": "#f59e0b", "KEEP": "#10b981",
        "REVIEW": "#3b82f6", "AUTOMATE": "#10b981", "DEFER": "#6366f1",
        "SKIP": "#4b5563", "APPROVAL_REQUIRED": "#06b6d4",
    }
    with pc3:
        if rec_counts:
            fig3 = make_pie(
                labels=list(rec_counts.keys()),
                values=list(rec_counts.values()),
                colors=[rec_colors_map.get(k, "#4b5563") for k in rec_counts.keys()],
                title="System Pipeline Recommendation Engine Output"
            )
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    thresholds = decision_engine.current_thresholds()
    cal_note = "system static defaults" if thresholds["is_default"] else "optimized runtime logistic matrix output"
    st.caption(f"Risk Separation Bounds Configuration Matrix: LOW Bracket < {thresholds['low_threshold']} · HIGH Bracket ≥ {thresholds['high_threshold']} ({cal_note})")

    # Storage Volume Progression Modeler Plotting Row
    fc_detail = st.session_state.get("forecast_detailed")
    if fc_detail:
        fc = fc_detail["forecast"]
        st.markdown('<div class="section-title">Storage Volume Volatility Projection</div>', unsafe_allow_html=True)
        if fc:
            fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)
            fcol1.metric("Current Partition Load", f"{fc['current_usage_pct']*100:.1f}%")
            fcol2.metric("Mean Scaling Speed Value", fc_detail["growth_rate_formatted"])
            d85 = fc["predictions_days"].get(0.85)
            fcol3.metric("Estimated Run-out Window to 85%", f"{d85:.0f} Days" if d85 else "Stable Boundary")
            d90 = fc["predictions_days"].get(0.90)
            fcol4.metric("Estimated Run-out Window to 90%", f"{d90:.0f} Days" if d90 else "Stable Boundary")
            d95 = fc["predictions_days"].get(0.95)
            fcol5.metric("Estimated Run-out Window to 95%", f"{d95:.0f} Days" if d95 else "Stable Boundary")
            
            history_rows = db.usage_series(st.session_state.target)
            if len(history_rows) >= 2:
                chart_df = pd.DataFrame([{
                    "Timeline Index Date": datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d"),
                    "Allocation Weight Metrics (GB)": r["used_bytes"] / (1024 ** 3),
                } for r in history_rows])
                chart_df = chart_df.drop_duplicates(subset="Timeline Index Date", keep="last").set_index("Timeline Index Date")
                st.line_chart(chart_df, height=180, color="#2f80ed")

            if fc_detail["cleanup_impact"]:
                st.markdown('<div class="section-title">Optimization Impact Modeling Analysis</div>', unsafe_allow_html=True)
                ci = fc_detail["cleanup_impact"]
                ci1, ci2, ci3, ci4 = st.columns(4)
                ci1.metric("Pre-Run Volumetric Footprint", ci["current_used_formatted"])
                ci2.metric("Purgeable Delta Pool", ci["reclaimable_formatted"])
                ci3.metric("Post-Run Expected Footprint", ci["estimated_after_formatted"])
                ci4.metric("Post-Run Target Ratio Scale", f"{ci['estimated_after_pct']*100:.0f}%")

        st.markdown('<div class="section-title">Engine Logic Advisory Dispatcher</div>', unsafe_allow_html=True)
        st.info(fc_detail["recommendation"])

    # High Level Behavioral Narrative Modules
    st.markdown('<div class="section-title">Data Environmental Archaeology Chronology</div>', unsafe_allow_html=True)
    archaeology_records = st.session_state.get("archaeology_records", [])
    stories = archaeology.build_stories(archaeology_records) if archaeology_records else []
    if stories:
        sc1, sc2 = st.columns(2)
        for idx, s in enumerate(stories[:6]):
            target_col = sc1 if idx % 2 == 0 else sc2
            with target_col:
                st.markdown(f"""
                <div class="narrative-block">
                  <div class="title">{s['headline']}</div>
                  <div class="desc">{s['detail']}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("System filesystem structures currently display regular uniform operational allocation profiles.")

# ---------------------------------------------------------------------------
# Targeted Candidate Evaluation Index Row
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Active Decision Stacks & Risk Allocation Map</div>', unsafe_allow_html=True)
results = st.session_state.get("results") or []
df = pd.DataFrame(results)

if not df.empty:
    display_df = pd.DataFrame([{
        "Target Node Element": r["File"],
        "Size Metrics": r["Size"],
        "Retention Stagnation (Days)": r["Age (days)"],
        "Lifecycle State": r["Classification"],
        "Operational Access Footprint": r.get("usage_profile") or "Stagnant",
        "Future Access Likelihood Ratio": (f"{r['future_usage_probability']*100:.1f}%"
                         if r.get("future_usage_probability") is not None else "0.0%"),
        "Calculated Risk Value Index": r.get("Risk Score", f"{r.get('risk_score', 0)} / 100"),
        "Risk Class Allocation Group": r.get("Risk", r.get("risk_tier", "MEDIUM")),
        "Action Strategy Selection": r.get("recommendation") or r.get("Action", r.get("action")),
        "Scoring Engine Evaluation Argument": r.get("recommendation_reason") or r.get("Reason", r.get("reason")),
    } for r in results])

    def risk_color_matrix(val):
        return {
            "LOW": "background-color: #0b251a; color: #10b981; font-weight: 600;",
            "MEDIUM": "background-color: #2a1b05; color: #f59e0b; font-weight: 600;",
            "HIGH": "background-color: #2d0d0d; color: #ef4444; font-weight: 600;",
        }.get(val, "")

    st.dataframe(display_df.style.map(risk_color_matrix, subset=["Risk Class Allocation Group"]), use_container_width=True, hide_index=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Export Machine Readable Logs (CSV)", data=csv_bytes,
                        file_name=f"aegisstore_telemetry_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv")

    # Interactive Deep Dive Inspector Module Configuration
    st.markdown('<div class="section-title">Deep Variable Engine Trace Inspector</div>', unsafe_allow_html=True)
    candidate_names = [r["File"] for r in results]
    selected_file = st.selectbox("Isolate telemetry nodes for trace parsing:", candidate_names, index=0)
    selected = next((r for r in results if r["File"] == selected_file), None)

    if selected:
        badge = risk_badge_html(selected.get("Risk", selected.get("risk_tier", "MEDIUM")))
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Assessed Core Element Risk Value", selected.get("Risk Score", f"{selected.get('risk_score', 0)} / 100"))
        col_b.metric("Engine Execution Assignment State", selected.get("Action", selected.get("action", "DEFER")))
        col_c.metric("Node Tree Existence Timeline Lifespan", f"{selected['Age (days)']} Days")

        st.markdown(f"**Target Host Resource Node Location Path Reference:** `{selected['Path']}` &nbsp;&nbsp; {badge}", unsafe_allow_html=True)

        if selected.get("recommendation"):
            st.markdown(f"**Predictive Model Synthesis Output Strategy:** `{selected['recommendation']}` — *{selected.get('recommendation_reason', '')}*")
        if selected.get("future_usage_probability") is not None:
            st.markdown(
                f"**Neural Access Pattern Forecast Ratio:** `{selected['future_usage_probability']*100:.1f}%` "
                f"({selected.get('future_usage_class', '—')}) — *{selected.get('future_usage_explanation', '')}*"
            )

        st.markdown("**Evaluated Pipeline Model Variable Weight Factors:**")
        factors = selected.get("factors", [])
        if factors:
            tags_html = "".join(f'<span class="micro-tag">{f}</span>' for f in factors)
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.caption("No custom scalar variables altered baseline mathematical matrix evaluations during pass loops.")
        st.markdown(f"**Execution Assignment Logic Argument:** {selected.get('Reason', selected.get('reason', ''))}")

        st.markdown("##### 🔄 Counterfactual Matrix Delta Modeling Simulation")
        try:
            from aegisstore import counterfactual
            original = next((c for c in scanner.scan_and_classify(st.session_state.target)
                              if str(c["path"]) == selected["Path"]), None)
            if original:
                cf = counterfactual.explain_age_change(
                    original, context.enrich(str(original["path"])), load, busy, days_delta=-7)
                cfc1, cfc2 = st.columns(2)
                cfc1.metric("Baseline Element Risk Index", f"{cf['current_score']} / 100")
                cfc2.metric("Counterfactual Shift Variance Simulation (-7 Days Age)", f"{cf['counterfactual_score']} / 100", delta=cf["delta"])
                st.info(f"💡 **Counterfactual Engine Extrapolation Model Analysis:** {cf['explanation']}")
                if cf["current_action"] != cf["counterfactual_action"]:
                    st.warning(f"Target logic path delta simulation flags macro state mutation: **{cf['current_action']}** → **{cf['counterfactual_action']}**")
        except Exception as e:
            st.caption(f"Counterfactual simulation engine failed execution bounds loops checks: {e}")

        st.markdown("##### 🎓 Human Feedback Input Tuning Vector Channels")
        fb1, fb2, fb3 = st.columns(3)
        rec_label = selected.get("recommendation") or selected.get("Action") or selected.get("action")
        if fb1.button("Confirm Strategy Vector", key=f"accept_{selected['Path']}", use_container_width=True):
            db.log_recommendation_feedback(selected["Path"], rec_label, selected.get("risk_score", 0),
                                            selected.get("future_usage_probability"), accepted=True)
            st.toast("Telemetry adjustment target vectors reinforced positively.")
        if fb2.button("Reject Strategy Vector", key=f"reject_{selected['Path']}", use_container_width=True):
            db.log_recommendation_feedback(selected["Path"], rec_label, selected.get("risk_score", 0),
                                            selected.get("future_usage_probability"), accepted=False)
            st.toast("Telemetry boundary weights penalization index updated.")
        fb3.caption(f"Currently staging {db.recommendation_feedback_count()} local user modification samples (Minimum threshold convergence limit requirement: 6+ nodes to recalculate core scalar weights).")

    st.markdown("##### 🧠 Global Retraining Core Optimization Module Trigger")
    rc1, rc2 = st.columns(2)
    if rc1.button("Retrain Core Models Now", use_container_width=True):
        import recalibrate as recalibrate_module
        result = recalibrate_module.recalibrate_from_feedback()
        if result["status"] == "recalibrated":
            st.success(
                f"Logistic model weights recalculated smoothly over {result['sample_count']} samples. "
                f"LOW boundary shifted: {result['old_thresholds']['low_threshold']} → {result['new_thresholds']['low_threshold']} · "
                f"HIGH boundary shifted: {result['old_thresholds']['high_threshold']} → {result['new_thresholds']['high_threshold']}"
            )
        elif result["status"] == "insufficient_data":
            st.info(f"Retraining requires data matrix growth. Minimal vector target variations missing (Staged: {result['sample_count']}/6 nodes).")
        elif result["status"] == "no_variation":
            st.info("The adjustment vector logs require balancing items. Please specify both affirmative alignments and standard rejections.")
    if rc2.button("Restore Engine Static Configuration Rules Defaults", use_container_width=True):
        import recalibrate as recalibrate_module
        recalibrate_module.reset_calibration()
        st.toast("Engine evaluation scalar boundaries fixed cleanly back to initial factory weights.")

else:
    st.info("Initialize a telemetry file system environment scan sequence to index system metrics targets.")

# ---------------------------------------------------------------------------
# Manual Pipeline Operations Executor Module Interface
# ---------------------------------------------------------------------------
if results:
    st.markdown('<div class="section-title">Manual System Action Stacks Dispatch Routing Channels</div>', unsafe_allow_html=True)
    auto_eligible = [r for r in results if r.get("Action") == "AUTOMATE" or r.get("action") == "AUTOMATE"]
    scheduled = [r for r in results if r.get("Action") == "SCHEDULE" or r.get("action") == "SCHEDULE"]
    approval = [r for r in results if r.get("Action") == "APPROVAL_REQUIRED" or r.get("action") == "APPROVAL_REQUIRED"]
    deferred = [r for r in results if r.get("Action") == "DEFER" or r.get("action") == "DEFER"]
    skipped = [r for r in results if r.get("Action") == "SKIP" or r.get("action") == "SKIP"]

    if auto_eligible:
        st.markdown(f'<div style="background: rgba(16, 185, 129, 0.05); padding:1rem; border-radius:6px; border:1px solid rgba(16,185,129,0.15); margin-bottom:1rem;"><strong>🟢 Safe Batch Processing Pool Available</strong><br/>{len(auto_eligible)} target workspace nodes meet verified compliance checks and can be instantly offloaded safely.</div>', unsafe_allow_html=True)
        col_batch, col_safety = st.columns(2)
        if col_batch.button("Execute Core Staged Batch Isolation Pipeline", type="primary", use_container_width=True):
            batch_candidates = [{"path": r["Path"], "reason": r.get("Reason", r.get("reason", ""))} for r in auto_eligible]
            batch_result = executor.batch_quarantine(batch_candidates, load, verify_safety=True)
            if batch_result["safety_cleared"]:
                if batch_result["executed"]:
                    st.success(f"Successfully processed {len(batch_result['executed'])} nodes. Cleared {batch_result['total_bytes_recovered'] / (1024**3):.2f} GB from partition maps.")
                if batch_result["failed"]:
                    st.warning(f"Engine validation errors tripped on {len(batch_result['failed'])} workspace targets during manipulation phases.")
            else:
                st.error("Infrastructure consistency monitor intercepted operation: Critical environment activity metrics boundaries breached.")
        col_safety.metric("Runtime Infrastructure Clearance Status Verification", "PASSED CLEAR" if not busy else "DEFERRED STALL")

    for action_name, items, guidance in [
        ("AUTOMATE", auto_eligible, "Instantly dispatchable targets verified against file access system anomalies maps."),
        ("SCHEDULE", scheduled, "Safe maintenance window deferred queue items awaiting system low activity flags."),
        ("APPROVAL_REQUIRED", approval, "Identity policy structural elements requiring elevation clear signatures."),
        ("DEFER", deferred, "Active IO block exceptions held until hardware resources normalize."),
        ("SKIP", skipped, "Active framework locks tripped. Operations hard-dropped to secure environment state integrity."),
    ]:
        if items:
            st.markdown(f"**Operational Pipeline Strategy Group: `{action_name}`** — *{guidance}*")
            for r in items:
                c1, c2 = st.columns(2)
                c1.markdown(f"<div style='font-family:"Geist Mono"; font-size:0.85rem; color:var(--text-main); padding: 4px 0;'>`{r['File']}` &middot; {r['Size']} &middot; <span style='color:var(--text-muted);'>{r.get('Reason', r.get('reason', ''))}</span></div>", unsafe_allow_html=True)
                if action_name == "AUTOMATE":
                    if c2.button("Quarantine Target", key=f"q_{r['Path']}", use_container_width=True):
                        try:
                            info = executor.quarantine_file(r["Path"], r.get("Reason", r.get("reason", "")))
                            st.toast(f"Node isolated securely. Checksum matching verification token: {info['integrity_verified']}")
                        except FileNotFoundError:
                            st.toast("Selected workspace object target node shifted scopes outside runtime window maps.")
                else:
                    c2.markdown("<div style='text-align:center; color:var(--text-muted); font-size:0.8rem; padding-top:4px;'>QUEUE HOLD</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Synthetic Operational Text Narrative Digest</div>', unsafe_allow_html=True)
    with st.spinner("Compiling context records analysis metrics..."):
        story = storage_story.generate_story(st.session_state.summary)
    st.info(story)

# ---------------------------------------------------------------------------
# Isolation Sandbox Staging Partition Management Console
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Secure Quarantine Sandbox State Registry Management</div>', unsafe_allow_html=True)
recovery_stats = executor.recovery_stats()
q1, q2, q3 = st.columns(3)
q1.metric("Quarantine Segment Object Count", recovery_stats["file_count"])
q2.metric("Staged Sandbox Storage Footprint", f"{recovery_stats['total_bytes'] / (1024**3):.2f} GB")
q3.metric("SHA-256 Bitwise Structural Match Ratio", f"{recovery_stats['integrity_ok']} / {recovery_stats['file_count']}")

quarantine_items = executor.list_quarantine(limit=50)
if quarantine_items:
    quar_df = pd.DataFrame([{
        "Isolated Filename Element": Path(item["original_path"]).name,
        "Historic Workspace Mount Absolute Path": item["original_path"],
        "Isolation Staging Strategy Rule Rationale": item["reason"],
        "Volumetric Size Scale (GB)": f"{item['size_bytes'] / (1024**3):.3f}",
        "Bitwise Signature Integrity State Status": "VERIFIED MATCH" if item["integrity_verified"] else "CORRUPTED ATTRIBUTE",
    } for item in quarantine_items])
    st.dataframe(quar_df, use_container_width=True, hide_index=True)

    selected_quar = st.selectbox("Isolate partitioned target references for recovery mapping loops:",
                                  options=[item["quarantine_path"] for item in quarantine_items],
                                  format_func=lambda p: Path(p).name)
    if selected_quar and st.button("Reverse Staging Strategy Partition Isolation Loops", use_container_width=True):
        try:
            restore_result = executor.undo_last(selected_quar)
            st.toast(f"Object node restored securely back to historic path mount: {restore_result['restored_to']}")
            st.rerun()
        except Exception as e:
            st.error(f"Reverse pipeline routing strategy execution trace threw kernel handling loop exceptions: {e}")
else:
    st.caption("Quarantine data structural index registry maps display an empty state configuration.")

# ---------------------------------------------------------------------------
# System Architecture Kernel Operations Audit Index
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Core Engine Event Trace Kernel Audit Journal</div>', unsafe_allow_html=True)
audit_rows = db.recent_audit(limit=15)
if audit_rows:
    audit_df = pd.DataFrame([dict(r) for r in audit_rows])
    st.dataframe(audit_df[["event_time", "action", "path", "reversible", "detail"]],
                 use_container_width=True, hide_index=True)
else:
    st.caption("Kernel operation audit tracks log array maps are currently reporting clean default initialization parameters.")

# ---------------------------------------------------------------------------
# Dynamic Chronological Process Scheduling Ledger
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Asynchronous Optimization Task Scheduling Frame Sequence Ledger</div>', unsafe_allow_html=True)
schedule_actions = {"DEFERRED", "RETRIED", "EXECUTED", "QUARANTINE"}
schedule_rows = [dict(r) for r in db.recent_audit(limit=50) if r["action"] in schedule_actions]
if schedule_rows:
    timeline = [{
        "Kernel Execution Clock Time": datetime.fromtimestamp(r["event_time"]).strftime("%H:%M:%S"),
        "Pipeline State Mutation Flags Event": "EXECUTED RUN" if r["action"] == "QUARANTINE" else r["action"],
        "Target Node Pointer Descriptor Location": r["path"],
        "Subsystem Execution Operational Code Context Argument": r["detail"],
    } for r in schedule_rows]
    st.dataframe(pd.DataFrame(timeline), use_container_width=True, hide_index=True)
else:
    st.caption("No internal scheduled loop sequences have initiated asynchronous deferred state changes yet in this execution epoch cycle context window.")
