"""AegisStore FastAPI backend. Run: uvicorn api:app --port 8001 --reload"""
import shutil
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from aegisstore import (
    archaeology, context, db, decision_engine, executor,
    future_usage_model, ml_training, predictor, recommendation_engine,
    safety_gate, scanner, storage_intelligence, storage_story, usage_intelligence,
)
from demo_setup import build_demo

app = FastAPI(title="AegisStore API", version="2.0.0")
db.init_db()
DEFAULT_TARGET = "./demo_disk"
_model = None
_last_summary = {}

def ensure_demo(target):
    t = Path(target)
    if not t.exists(): build_demo(t)
    total, used, _ = shutil.disk_usage(t)
    db.log_usage(target, used, total)
    if predictor.forecast(target, min_points=3) is None:
        predictor.seed_synthetic_history(target, total, current_used_bytes=used, daily_growth_gb=1.8, days_back=14)

ensure_demo(DEFAULT_TARGET)

def get_model():
    global _model
    if _model is None:
        X, y = ml_training.generate_training_data(samples=3000, seed=42)
        _model = future_usage_model.train_model(X, y)
    return _model

def hb(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

@app.get("/api/system")
def system():
    load = safety_gate.read_system_load(sample_seconds=0.3)
    busy = safety_gate.is_system_busy(load)
    return {"cpu":round(load["cpu_percent"],1),"ram":round(load["memory_percent"],1),
            "io":round(load["io_wait_percent"],1),"read":round(load["disk_read_mb_s"],2),
            "write":round(load["disk_write_mb_s"],2),"state":load.get("state","NORMAL"),"busy":busy}

@app.get("/api/thresholds")
def thresholds():
    return decision_engine.current_thresholds()

@app.post("/api/scan")
def scan(target: str = DEFAULT_TARGET):
    global _last_summary
    t = Path(target)
    if not t.exists(): build_demo(t)
    total, used, _ = shutil.disk_usage(t)
    db.log_usage(target, used, total)
    load = safety_gate.read_system_load(sample_seconds=0.3)
    busy = safety_gate.is_system_busy(load)
    records  = scanner.scan_and_classify(target)
    analyzed = usage_intelligence.analyze_records(records)
    model    = get_model()
    predicted = [future_usage_model.predict_record(model, r) for r in analyzed]
    candidates, reclaimable = scanner.reclaimable_summary(records)
    cpaths = {str(c["path"]) for c in candidates}
    crecs  = sorted([r for r in predicted if str(r["path"]) in cpaths], key=lambda r: r["size_bytes"], reverse=True)[:20]
    rows = []
    for r in crecs:
        ctx = context.enrich(str(r["path"]))
        dec = decision_engine.assess(r, ctx, load, busy)
        rec = recommendation_engine.recommend({**r, **ctx})
        if dec["action"] == "DEFER":
            db.log_schedule_event(r["path"], "DEFERRED", load, reason=dec["reason"])
        cid = db.save_candidate(r)
        db.save_decision(cid, {**ctx,"cpu_percent":load["cpu_percent"],"io_wait_percent":load["io_wait_percent"],**dec})
        rows.append({"id":cid,"name":r["path"].name,"path":str(r["path"]),
            "size":hb(r["size_bytes"]),"size_bytes":r["size_bytes"],"age_days":round(r["age_days"],1),
            "cls":r["classification"],"confidence":round(r["confidence"]*100),
            "risk":dec["risk_tier"],"score":dec["risk_score"],"action":dec["action"],
            "reason":dec["reason"],"factors":dec.get("factors",[]),
            "future_pct":round((r.get("future_usage_probability") or 0)*100,1),
            "future_cls":r.get("future_usage_class",""),"future_exp":r.get("future_usage_explanation",""),
            "recommendation":rec.get("recommendation",""),"recommendation_reason":rec.get("recommendation_reason",""),
            "pkg_owned":ctx.get("package_owned",False),"git_tracked":ctx.get("git_tracked",False),
            "active_proc":ctx.get("active_process",False)})
    stories = [{"headline":s["headline"],"detail":s["detail"]} for s in archaeology.build_stories(analyzed)]
    fc_det  = storage_intelligence.storage_forecast_detailed(target, reclaimable_bytes=reclaimable)
    thr     = decision_engine.current_thresholds()
    auto    = [r for r in rows if r["action"]=="AUTOMATE"]
    defer   = [r for r in rows if r["action"]=="DEFER"]
    summary = {"total_candidates":len(records),"reclaimable":hb(reclaimable),
        "reclaimable_bytes":reclaimable,"disk_used":hb(used),"disk_total":hb(total),
        "disk_pct":round(used/total*100,1),"automate_count":len(auto),"deferred_count":len(defer),
        "top_reason":"cold/redundant data","avg_confidence":0.85,
        "system":{"cpu":round(load["cpu_percent"],1),"busy":busy,"state":load.get("state","NORMAL")},
        "forecast":fc_det,"stories":stories,"thresholds":thr}
    _last_summary = summary
    return {"candidates":rows,"summary":summary}

@app.get("/api/forecast")
def forecast(target: str = DEFAULT_TARGET):
    fc      = predictor.forecast(target)
    history = db.usage_series(target)
    return {"forecast":fc,"history":[{"date":str(r["timestamp"]),"gb":round(r["used_bytes"]/(1024**3),2)} for r in history]}

@app.get("/api/counterfactual")
def counterfactual_api(path: str, target: str = DEFAULT_TARGET):
    from aegisstore import counterfactual as cf_mod
    load = safety_gate.read_system_load(sample_seconds=0.2)
    busy = safety_gate.is_system_busy(load)
    records = scanner.scan_and_classify(target)
    orig = next((r for r in records if str(r["path"])==path), None)
    if not orig: raise HTTPException(404, "File not found in scan")
    ctx = context.enrich(path)
    try:
        return cf_mod.explain_age_change(orig, ctx, load, busy, days_delta=-7)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/story")
def story_api():
    if not _last_summary: raise HTTPException(400, "Run a scan first")
    try:
        return {"story": storage_story.generate_story(_last_summary)}
    except:
        return {"story": f"Storage has {_last_summary.get('total_candidates','?')} candidates, {_last_summary.get('reclaimable','?')} reclaimable."}

@app.get("/api/audit")
def audit(limit: int = 20):
    return [dict(r) for r in db.recent_audit(limit=limit)]

@app.get("/api/schedule")
def schedule():
    sa = {"DEFERRED","RETRIED","EXECUTED","QUARANTINE"}
    return [dict(r) for r in db.recent_audit(limit=50) if r["action"] in sa]

@app.get("/api/quarantine")
def quarantine_list():
    return {"stats":executor.recovery_stats(),"files":executor.list_quarantine(limit=50)}

class QuarantineReq(BaseModel):
    path: str
    reason: str = "Manual quarantine"

@app.post("/api/quarantine")
def quarantine_file(req: QuarantineReq):
    try:
        return {"ok":True, **executor.quarantine_file(req.path, req.reason)}
    except FileNotFoundError:
        raise HTTPException(404, "File not found or already quarantined")
    except Exception as e:
        raise HTTPException(500, str(e))

class BatchReq(BaseModel):
    candidates: List[dict]

@app.post("/api/batch_quarantine")
def batch_quarantine(req: BatchReq):
    load = safety_gate.read_system_load(sample_seconds=0.2)
    return executor.batch_quarantine(req.candidates, load, verify_safety=True)

class RestoreReq(BaseModel):
    quarantine_path: str

@app.post("/api/restore")
def restore(req: RestoreReq):
    try:
        return {"ok":True, **executor.undo_last(req.quarantine_path)}
    except Exception as e:
        raise HTTPException(500, str(e))

class FeedbackReq(BaseModel):
    path: str
    recommendation: str
    risk_score: int
    future_prob: float = 0.0
    accepted: bool = True

@app.post("/api/feedback")
def feedback(req: FeedbackReq):
    db.log_recommendation_feedback(req.path, req.recommendation, req.risk_score, req.future_prob, req.accepted)
    return {"ok":True,"total":db.recommendation_feedback_count(),"thresholds":decision_engine.current_thresholds()}

@app.get("/api/feedback_count")
def feedback_count():
    return {"count":db.recommendation_feedback_count(),"thresholds":decision_engine.current_thresholds()}

@app.post("/api/recalibrate")
def recalibrate():
    import recalibrate as rm
    return rm.recalibrate_from_feedback()

@app.post("/api/reset_calibration")
def reset_calibration():
    import recalibrate as rm
    rm.reset_calibration()
    return {"ok":True,"message":"Reset to defaults (LOW<31, HIGH≥66)"}

@app.post("/api/reset")
def reset(target: str = DEFAULT_TARGET):
    t = Path(target)
    if t.exists(): shutil.rmtree(t)
    for p in [Path("quarantine"),Path("aegisstore.db"),Path("calibration.json")]:
        if p.exists(): (shutil.rmtree(p) if p.is_dir() else p.unlink())
    db.init_db(); ensure_demo(target)
    return {"ok":True}

static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    index = static_dir / "index.html"
    return FileResponse(index) if index.exists() else JSONResponse({"message":"Place index.html in /static/"})

@app.get("/health")
def health():
    return {"status":"ok","version":"2.0.0"}
