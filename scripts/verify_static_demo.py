"""
Compare exported static recs against the real RetrievalEngine.recommend_for_climb.

Usage:
  python scripts/verify_static_demo.py [--engine-repo .] [--data data] [--n 100]

Exact ties and float-noise near-ties can legitimately swap items, so when strict
order differs the check falls back to per-rank score agreement within 1e-4.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from export_static_demo import TOP_K, load_engine_module  # noqa: E402


def groups(items):
    """[(id, score, ...)] -> list of frozensets of ids, grouped by equal (rounded) score."""
    out, cur, last = [], set(), None
    for it in items:
        s = round(it[1], 4)
        if last is not None and s != last:
            out.append(frozenset(cur))
            cur = set()
        cur.add(it[0])
        last = s
    if cur:
        out.append(frozenset(cur))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-repo", default=".")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    retrieval = load_engine_module(args.engine_repo)
    db_dir = os.path.join(args.engine_repo, "server", "db")

    class _Stub:  # SentenceTransformer stand-in
        def __init__(self, *a, **k):
            pass
    retrieval.SentenceTransformer = _Stub
    engine = retrieval.RetrievalEngine(
        os.path.join(db_dir, "db.sqlite"), os.path.join(db_dir, "subpattern_index"))

    # Which climbs were exported?
    exported = {}
    rec_dir = os.path.join(args.data, "recs")
    for fn in os.listdir(rec_dir):
        with open(os.path.join(rec_dir, fn), encoding="utf-8") as f:
            exported.update(json.load(f))

    # Outliers: climbs with the most windows.
    hubs = [r[0] for r in engine.db.execute(
        "SELECT climb_id FROM subpattern_occurrences GROUP BY climb_id "
        "ORDER BY COUNT(*) DESC LIMIT 5")]
    rng = random.Random(args.seed)
    ids = sorted(exported)
    sample = [c for c in hubs if c in exported] + rng.sample(ids, min(args.n, len(ids)))

    bad = {"order": 0, "ties_ok": 0, "hard": 0}
    examples = []
    for cid in sample:
        for mode in ("similar", "opposite"):
            ref = engine.recommend_for_climb(cid, mode=mode, top_k=TOP_K)
            got = exported[cid][mode]
            want = [[m["climb_id"], m["score"], int(m["is_mirrored"]), m["matched_window_count"]]
                    for m in ref]
            same_order = (
                [g[0] for g in got] == [w[0] for w in want]
                and all(abs(g[1] - w[1]) < 1e-4 for g, w in zip(got, want))
                and all(g[2:] == w[2:] for g, w in zip(got, want))
            )
            if same_order:
                continue
            bad["order"] += 1
            # Exact ties, or float-noise near-ties (batched matmul vs the engine's
            # matvec differ by ~1e-6): every rank's score still agrees to 1e-4.
            if all(abs(g[1] - w[1]) < 1e-4 for g, w in zip(got, want)):
                bad["ties_ok"] += 1
            else:
                bad["hard"] += 1
                if len(examples) < 3:
                    examples.append((cid, mode, want, got))

    total = len(sample) * 2
    print(f"checked {total} (climb, mode) pairs: "
          f"{total - bad['order']} exact, {bad['ties_ok']} tie-reorder only, {bad['hard']} MISMATCH")
    for cid, mode, want, got in examples:
        print(f"\n{cid} {mode}\n engine: {want}\n export: {got}")
    engine.close()
    sys.exit(1 if bad["hard"] else 0)


if __name__ == "__main__":
    main()
