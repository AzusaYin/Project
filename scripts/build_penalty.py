from pathlib import Path
import json, collections

# 让路径相对于项目根目录，而不是当前脚本文件夹
ROOT = Path(__file__).resolve().parent.parent
FB = ROOT / "data/feedback/feedback.jsonl"
OUT = ROOT / "data/feedback/penalty.json"

cnt = collections.Counter()
if not FB.exists():
    print(f"[WARN] Feedback file not found: {FB}")
    raise SystemExit(0)

for line in FB.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        r = json.loads(line)
    except Exception as e:
        print(f"[WARN] bad line: {e}")
        continue
    if r.get("label") != "down":
        continue
    for c in r.get("citations", []):
        key = (c.get("file"), c.get("page"))
        cnt[key] += 1

# 阈值：>=3 次 down 记入惩罚
pen = {f"{k[0]}::{k[1]}": min(1.0, 0.15 + 0.05*v) for k,v in cnt.items() if v >= 3}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(pen, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] Penalized pages: {len(pen)} → {OUT}")