"""
archaeology.py — "Digital archaeology" summary mode.

Takes the same classified/enriched records the dashboard already computes and
groups them into a handful of plain-English stories a non-technical judge can
read in one glance, e.g.:

  "14.2 MB across 2 duplicate files in datasets/ — safe to consolidate."
  "31.0 MB of inactive data in build/, untouched for 180+ days."

This module only reads data that scanner/usage_intelligence/recommendation_engine
already computed — it never re-scans the filesystem and never touches files.
"""
from collections import defaultdict
from pathlib import Path


def _human(n_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def _bucket_for(record: dict) -> str:
    """Which story bucket a file belongs in. Duplicate status takes priority
    over usage profile, since 'duplicate' is the more actionable story."""
    if record.get("is_duplicate") or record.get("duplicate_of"):
        return "Duplicate"
    profile = str(record.get("usage_profile") or record.get("classification") or "Unknown").lower()
    if "inactive" in profile or "cold" in profile:
        return "Inactive / Cold"
    if "warm" in profile:
        return "Moderately used"
    if "hot" in profile:
        return "Actively used"
    return "Unclassified"

# Only these buckets represent genuine optimization opportunities worth
# narrating as a "story" — Hot/actively-used files are deliberately excluded.
_STORY_WORTHY_BUCKETS = {"Duplicate", "Inactive / Cold"}


def build_stories(records: list[dict], min_group_bytes: int = 1024 * 1024) -> list[dict]:
    """
    Groups records by (parent directory, bucket) and returns one story per
    group above min_group_bytes, sorted largest-impact first.

    Returns: [{"headline": str, "detail": str, "bytes": int, "file_count": int,
               "bucket": str, "directory": str}]
    """
    groups = defaultdict(list)
    for r in records:
        bucket = _bucket_for(r)
        if bucket not in _STORY_WORTHY_BUCKETS:
            continue
        directory = str(Path(r["path"]).parent.name) or "/"
        groups[(directory, bucket)].append(r)

    stories = []
    for (directory, bucket), items in groups.items():
        total_bytes = sum(int(i.get("size_bytes", 0)) for i in items)
        if total_bytes < min_group_bytes:
            continue

        count = len(items)
        if bucket == "Duplicate":
            headline = f"{_human(total_bytes)} across {count} duplicate file{'s' if count != 1 else ''} in {directory}/"
            detail = "These files are byte-for-byte identical to other files already on disk — safe to consolidate."
        else:
            ages = [i.get("days_since_access", i.get("age_days", 0)) for i in items]
            max_age = max(ages) if ages else 0
            headline = f"{_human(total_bytes)} of inactive data in {directory}/, untouched for {max_age:.0f}+ days"
            detail = "Little or no recent access activity — a candidate for archiving or cleanup, pending review."

        stories.append({
            "headline": headline,
            "detail": detail,
            "bytes": total_bytes,
            "file_count": count,
            "bucket": bucket,
            "directory": directory,
        })

    stories.sort(key=lambda s: s["bytes"], reverse=True)
    return stories
