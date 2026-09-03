#!/usr/bin/env python3

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from aegisstore import scanner
from pathlib import Path

records = scanner.scan_and_classify('./demo_disk')
print(f'Total files scanned: {len(records)}')
for r in records[:10]:
    print(f'  {r["path"].name}: {r["classification"]}, age={r["age_days"]:.1f} days')
    
candidates, total = scanner.reclaimable_summary(records)
print(f'Candidates for cleanup: {len(candidates)}, total bytes: {total}')
print("\nCandidate files:")
for c in candidates:
    print(f'  {c["path"].name}: {c["classification"]}')
