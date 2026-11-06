"""Attribute GPU kernel time in an nsys sqlite export to the innermost NVTX range of its launch.

Usage: python bench/nsys_attr.py <trace.sqlite> [--by kernel|range] [--top N]
Prints, per NVTX range path (outer;inner), total kernel time and number of kernels,
plus the top kernels inside each range.
"""
import argparse
import bisect
import sqlite3
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("db")
ap.add_argument("--top", type=int, default=6)
ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--match", default="", help="only ranges whose path contains this")
args = ap.parse_args()
con = sqlite3.connect(args.db)
q = lambda s: con.execute(s).fetchall()
strings = dict(q("select id, value from StringIds"))
nv = q("select start, end, coalesce(text, ''), textId, globalTid from NVTX_EVENTS where end is not null and eventType=59")
by_tid = defaultdict(list)
for s, e, t, tid_, g in nv:
    by_tid[g].append((s, e, t or strings.get(tid_, "?")))
for g in by_tid:
    by_tid[g].sort()
rt = {c: (s, g) for s, g, c in q("select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
kern = q("select k.start, k.end, k.correlationId, k.shortName from CUPTI_ACTIVITY_KIND_KERNEL k")


# Innermost-first range stack at each runtime call, by sweeping calls and ranges per thread.
calls_by_tid = defaultdict(list)
for c, (t, g) in rt.items():
    calls_by_tid[g].append((t, c))
path_of_call = {}
for g, calls in calls_by_tid.items():
    calls.sort()
    rs = by_tid.get(g, [])
    stack, i = [], 0
    for t, c in calls:
        while i < len(rs) and rs[i][0] <= t:
            stack.append(rs[i])
            i += 1
        stack = [r for r in stack if r[1] >= t]
        path_of_call[c] = tuple(r[2] for r in stack)


cache = {}
tot = defaultdict(float)
cnt = defaultdict(int)
kin = defaultdict(lambda: defaultdict(float))
alltot = 0
for s, e, c, sn in kern:
    d = (e - s) / 1e9
    alltot += d
    r = rt.get(c)
    p = path_of_call.get(c, ("<no launch>",))
    p = p[: args.depth]
    if args.match and args.match not in ";".join(p):
        continue
    key = ";".join(p) if p else "<none>"
    tot[key] += d
    cnt[key] += 1
    kin[key][strings.get(sn, sn)] += d
calls = defaultdict(int)
for g, rs in by_tid.items():
    for _, _, name in rs:
        calls[name] += 1
print(f"total kernel time {alltot:.3f}s")
for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
    if v < 0.02:
        continue
    n = calls.get(k.split(";")[-1], 0) or 1
    print(f"{v:8.3f}s {100*v/alltot:5.1f}% {cnt[k]:7d} kernels  {n:6d} range calls  {1e3*v/n:8.3f} ms/call  {k}")
    for kn, kv in sorted(kin[k].items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"            {kv:7.3f}s {1e3*kv/n:8.3f} ms/call  {kn[:90]}")
