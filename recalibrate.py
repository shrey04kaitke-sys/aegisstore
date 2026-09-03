"""
recalibrate.py — Feedback-driven recalibration loop.

Reads every recommendation_feedback row (a user accepting or rejecting a past
recommendation), fits a simple logistic regression mapping risk_score -> P(accept),
and derives new LOW/HIGH risk-tier thresholds from where that curve crosses 0.5
and 0.85. The result is written to calibration.json, which decision_engine.py
reads on every future scoring call.

This is a real learning loop, not a cosmetic one: after recalibration, the
SAME file with the SAME risk score can land in a different risk tier, because
the boundary itself moved based on what the user actually accepted or rejected.

Usage:
    python3 recalibrate.py            # run recalibration from logged feedback
    python3 recalibrate.py --reset    # delete calibration.json, revert to defaults
"""
import json
import sys
from pathlib import Path

import numpy as np

from aegisstore import db

CALIBRATION_PATH = Path(__file__).parent / "calibration.json"
MIN_FEEDBACK_ROWS = 6  # below this, a logistic fit is unstable/meaningless


def recalibrate_from_feedback(min_rows: int = MIN_FEEDBACK_ROWS) -> dict:
    """
    Returns a result dict describing what happened:
      {"status": "insufficient_data" | "no_variation" | "recalibrated",
       "sample_count": int,
       "old_thresholds": {...}, "new_thresholds": {...} (if recalibrated)}
    Never raises — always safe to call from the dashboard.
    """
    from aegisstore.decision_engine import current_thresholds

    old = current_thresholds()
    rows = db.recommendation_feedback_rows(limit=1000)

    if len(rows) < min_rows:
        return {"status": "insufficient_data", "sample_count": len(rows),
                "old_thresholds": old, "new_thresholds": old}

    scores = np.array([r["risk_score"] for r in rows], dtype=float)
    accepted = np.array([r["accepted"] for r in rows], dtype=int)

    # Need both classes present (at least one accept AND one reject) to fit anything.
    if len(set(accepted.tolist())) < 2:
        return {"status": "no_variation", "sample_count": len(rows),
                "old_thresholds": old, "new_thresholds": old}

    try:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(scores.reshape(-1, 1), accepted)

        # Find the score where predicted P(accept) crosses 0.5 and 0.15 --
        # i.e. "users start rejecting" and "users almost always reject."
        probe = np.linspace(0, 100, 401).reshape(-1, 1)
        probs = model.predict_proba(probe)[:, 1]

        below_50 = probe[probs >= 0.5]
        new_low = float(below_50.max()) if len(below_50) else old["low_threshold"]

        below_15 = probe[probs >= 0.15]
        new_high = float(below_15.max()) if len(below_15) else old["high_threshold"]

        # Keep thresholds sane and ordered regardless of what the fit produced.
        new_low = max(5.0, min(new_low, 60.0))
        new_high = max(new_low + 10.0, min(new_high, 95.0))

        CALIBRATION_PATH.write_text(json.dumps({
            "low_threshold": round(new_low, 1),
            "high_threshold": round(new_high, 1),
            "sample_count": len(rows),
        }, indent=2))

        new = {"low_threshold": round(new_low, 1), "high_threshold": round(new_high, 1),
               "is_default": False}
        return {"status": "recalibrated", "sample_count": len(rows),
                "old_thresholds": old, "new_thresholds": new}
    except Exception as e:
        return {"status": "error", "sample_count": len(rows), "error": str(e),
                "old_thresholds": old, "new_thresholds": old}


def reset_calibration():
    if CALIBRATION_PATH.exists():
        CALIBRATION_PATH.unlink()


def main():
    if "--reset" in sys.argv:
        reset_calibration()
        print("Calibration reset to defaults (LOW < 31, HIGH >= 66).")
        return

    result = recalibrate_from_feedback()
    print(f"Status: {result['status']}  (from {result['sample_count']} feedback samples)")
    if result["status"] == "recalibrated":
        print(f"  Old thresholds: LOW < {result['old_thresholds']['low_threshold']}, "
              f"HIGH >= {result['old_thresholds']['high_threshold']}")
        print(f"  New thresholds: LOW < {result['new_thresholds']['low_threshold']}, "
              f"HIGH >= {result['new_thresholds']['high_threshold']}")
    elif result["status"] == "insufficient_data":
        print(f"  Need at least {MIN_FEEDBACK_ROWS} feedback samples "
              f"(accept/reject clicks in the dashboard) before recalibrating.")
    elif result["status"] == "no_variation":
        print("  All feedback so far is the same direction (all accepts or all "
              "rejects) — need at least one of each to fit a boundary.")


if __name__ == "__main__":
    main()
