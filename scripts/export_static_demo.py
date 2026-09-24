"""
Batch-precompute RetrievalEngine.recommend_for_climb() for every indexed climb
(both modes, fixed TOP_K) and write static JSON shards for the demo client.

Output layout (under --out, default data/):
  names.json            [[climb_id, name, setter, created_at, [[angle, grade], ...]], ...]
  recs/<shard>.json     {climb_id: {"similar": [[id, score, mirrored, windows], ...],
                                    "opposite": [...]}}
  holds/<shard>.json    {climb_id: [[x, y, role_id], ...]}
  meta.json             {"top_k": TOP_K, ...}
shard = first 2 chars of the climb uuid, lowercased (ids are mixed case; case-insensitive filesystems would collide).

Two resumable stages (work files live in --work):
  1. search:   per sequence_length, blocked matmul of every window against the
               pool, tie-faithful per-window top-k (same order the engine's
               stable sort produces). Checkpoints one .npz per query block.
  2. assemble: per-climb aggregation exactly as recommend_for_climb(), then
               shard writing.

Usage:
  python scripts/export_static_demo.py [--engine-repo .] [--limit N]
"""

import argparse
import json
import math
import os
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np

TOP_K = 5
BLOCK = 512
THREADS = 8


def load_engine_module(engine_repo):
    # The embedding model is never used at search time; stub the heavy/absent
    # imports so retrieval.py imports without torch.
    for name, attrs in (("sentence_transformers", {"SentenceTransformer": object}),
                        ("rapidfuzz", {"fuzz": None, "process": None})):
        try:
            __import__(name)
        except ImportError:
            mod = types.ModuleType(name)
            mod.__dict__.update(attrs)
            sys.modules[name] = mod
    sys.path.insert(0, os.path.abspath(engine_repo))
    from server.generate_training_data_v2 import retrieval
    return retrieval


class Engine:
    """Bare DB/index access; mirrors what RetrievalEngine.__init__ sets up, minus the model."""

    def __init__(self, retrieval, engine_repo):
        import sqlite3
        db_dir = os.path.join(engine_repo, "server", "db")
        self.db_path = os.path.join(db_dir, "db.sqlite")
        prefix = os.path.join(db_dir, "subpattern_index")
        with open(prefix + ".meta.json") as f:
            meta = json.load(f)
        dtype = np.float16 if meta["dtype"] == "float16" else np.float32
        self.vectors = np.memmap(prefix + ".dat", dtype=dtype, mode="r",
                                 shape=(meta["count"], meta["dim"]))
        self.db = sqlite3.connect(self.db_path)
        self.map_db = sqlite3.connect(prefix + ".sqlite")
        self.weight = {}
        for seq_len, key, df in self.db.execute("""
            SELECT sequence_length, canonical_key, COUNT(DISTINCT climb_id)
            FROM subpattern_occurrences GROUP BY sequence_length, canonical_key
        """):
            self.weight[(seq_len, key)] = 1.0 / (1.0 + math.log(df))
        # Reuse the engine's own row-level helpers for enrichment/holds.
        self._retrieval = retrieval

    def enrich_angles(self, climb_id):
        return self._retrieval.RetrievalEngine._enrich_angles(self, climb_id)

    def get_climb_holds(self, climb_id):
        return self._retrieval.RetrievalEngine.get_climb_holds(self, climb_id)

    def load_pool(self, L):
        """Same SQL (and thus row order) as _load_candidate_pool, minus the exclusion."""
        rows = self.db.execute(
            """
            SELECT id, climb_id, climb_name, is_mirrored, canonical_key
            FROM subpattern_occurrences
            WHERE sequence_length = ?
            """, (L,)).fetchall()
        row_index = dict(self.map_db.execute(
            "SELECT occurrence_id, row_index FROM embedding_index").fetchall())
        return [r for r in rows if r[0] in row_index], row_index


def select_topk(scores, k):
    """
    scores: (b, N) float32 where higher == better and excluded cols are -inf.
    Returns (b, k) column indices of the top-k, ties resolved toward lower
    column (pool order), matching a stable sort. Order within the k is NOT
    final; caller sorts.
    """
    b = scores.shape[0]
    part = np.argpartition(scores, -k, axis=1)[:, -k:]
    kth = np.take_along_axis(scores, part, axis=1).min(axis=1)
    n_gt = (scores > kth[:, None]).sum(axis=1)
    n_eq = (scores == kth[:, None]).sum(axis=1)
    for r in np.nonzero(n_eq > (k - n_gt))[0]:
        gt = np.nonzero(scores[r] > kth[r])[0]
        eq = np.nonzero(scores[r] == kth[r])[0][: k - len(gt)]
        part[r] = np.concatenate([gt, eq])
    return part


def search_block(qvecs, own_cols, pool_vecs, weights, k):
    """Returns (sim_cols, sim_scores, opp_cols, opp_scores), each (b, k), final order."""
    raw = qvecs @ pool_vecs.T  # (b, N) float32
    rows = np.concatenate([np.full(len(c), i) for i, c in enumerate(own_cols)])
    cols = np.concatenate(own_cols)

    def work(sl):
        r = raw[sl]
        sub_rows = rows[(rows >= sl.start) & (rows < sl.stop)] - sl.start
        sub_cols = cols[(rows >= sl.start) & (rows < sl.stop)]
        sim = r * weights[None, :]
        sim[sub_rows, sub_cols] = -np.inf
        sim_c = select_topk(sim, k)
        sim_s = np.take_along_axis(sim, sim_c, axis=1)
        order = np.lexsort((sim_c, -sim_s), axis=1)
        sim_c = np.take_along_axis(sim_c, order, axis=1)
        sim_s = np.take_along_axis(sim_s, order, axis=1)

        neg = -r
        neg[sub_rows, sub_cols] = -np.inf
        opp_c = select_topk(neg, k)
        opp_s = -np.take_along_axis(neg, opp_c, axis=1)
        order = np.lexsort((opp_c, opp_s), axis=1)
        opp_c = np.take_along_axis(opp_c, order, axis=1)
        opp_s = np.take_along_axis(opp_s, order, axis=1)
        return sim_c, sim_s, opp_c, opp_s

    b = raw.shape[0]
    step = max(1, math.ceil(b / THREADS))
    slices = [slice(i, min(i + step, b)) for i in range(0, b, step)]
    with ThreadPoolExecutor(THREADS) as ex:
        parts = list(ex.map(work, slices))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def stage_search(eng, work_dir, limit_climbs):
    os.makedirs(work_dir, exist_ok=True)
    for L in (3, 4, 5):
        pool_rows, row_index = eng.load_pool(L)
        n = len(pool_rows)
        print(f"[search] L={L}: pool {n}", flush=True)
        vec_idx = np.array([row_index[r[0]] for r in pool_rows])
        # Gather in sorted order for memmap locality, then restore pool order.
        order = np.argsort(vec_idx)
        pool_vecs = np.empty((n, eng.vectors.shape[1]), dtype=np.float32)
        pool_vecs[order] = np.asarray(eng.vectors[vec_idx[order]]).astype(np.float32)
        weights = np.array([eng.weight.get((L, r[4]), 1.0) for r in pool_rows], dtype=np.float32)

        by_climb = {}
        for i, r in enumerate(pool_rows):
            by_climb.setdefault(r[1], []).append(i)
        by_climb = {c: np.array(v) for c, v in by_climb.items()}

        if limit_climbs is None:
            queries = np.arange(n)
        else:
            queries = np.concatenate([by_climb[c] for c in limit_climbs if c in by_climb])
            queries.sort()

        t0 = time.time()
        for bi, start in enumerate(range(0, len(queries), BLOCK)):
            path = os.path.join(work_dir, f"L{L}_b{start:07d}.npz")
            if os.path.exists(path):
                continue
            q = queries[start:start + BLOCK]
            own = [by_climb[pool_rows[i][1]] for i in q]
            sc, ss, oc, os_ = search_block(pool_vecs[q], own, pool_vecs, weights, TOP_K)
            tmp = path + ".tmp.npz"
            np.savez(tmp, q=q, sim_c=sc, sim_s=ss, opp_c=oc, opp_s=os_)
            os.replace(tmp, path)
            if bi % 10 == 0:
                done = start + len(q)
                el = time.time() - t0
                print(f"[search] L={L} {done}/{len(queries)}  {el:.0f}s "
                      f"(eta {el / done * (len(queries) - done):.0f}s)", flush=True)


def stage_assemble(eng, work_dir, out_dir, limit_climbs):
    import glob
    # All occurrences, grouped by climb in id (SCAN) order -- same order as
    # recommend_for_climb()'s reference_windows query.
    occ = eng.db.execute(
        "SELECT id, climb_id, sequence_length FROM subpattern_occurrences").fetchall()
    windows_by_climb = {}
    for oid, cid, L in occ:
        windows_by_climb.setdefault(cid, []).append((oid, L))

    pools, results = {}, {}
    pos_of = {}  # occurrence id -> (L, pool position)
    for L in (3, 4, 5):
        rows, _ = eng.load_pool(L)
        pools[L] = rows
        for i, r in enumerate(rows):
            pos_of[r[0]] = (L, i)
        res = {}
        for path in sorted(glob.glob(os.path.join(work_dir, f"L{L}_b*.npz"))):
            z = np.load(path)
            for j, qi in enumerate(z["q"]):
                res[int(qi)] = (z["sim_c"][j], z["sim_s"][j], z["opp_c"][j], z["opp_s"][j])
        results[L] = res

    climb_ids = sorted(windows_by_climb)
    if limit_climbs is not None:
        climb_ids = limit_climbs

    angle_cache = {}

    def angles(cid):
        if cid not in angle_cache:
            angle_cache[cid] = [[a["angle"], a["grade"]] for a in eng.enrich_angles(cid)]
        return angle_cache[cid]

    recs_by_shard = {}
    for n, cid in enumerate(climb_ids):
        entry = {}
        for mode, ci, si in (("similar", 0, 1), ("opposite", 2, 3)):
            better = (lambda a, b: a > b) if mode == "similar" else (lambda a, b: a < b)
            best, counts = {}, {}
            for oid, L in windows_by_climb[cid]:
                _, pos = pos_of[oid]
                r = results[L][pos]
                for col, score in zip(r[ci], r[si]):
                    row = pools[L][col]
                    m = (row[1], row[2], bool(row[3]), float(score))
                    if m[0] not in best or better(m[3], best[m[0]][3]):
                        best[m[0]] = m
                    counts[m[0]] = counts.get(m[0], 0) + 1
            ranked = sorted(best.values(), key=lambda m: m[3], reverse=(mode == "similar"))[:TOP_K]
            entry[mode] = [[m[0], round(m[3], 6), int(m[2]), counts[m[0]]] for m in ranked]
        recs_by_shard.setdefault(cid[:2].lower(), {})[cid] = entry
        if n % 2000 == 0:
            print(f"[assemble] {n}/{len(climb_ids)}", flush=True)

    write_dir = lambda name: os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    write_dir("recs"), write_dir("holds")
    for shard, payload in recs_by_shard.items():
        dump(os.path.join(out_dir, "recs", f"{shard}.json"), payload)

    # holds for every indexed climb (matches are drawn on the board too)
    holds_by_shard = {}
    for cid in (sorted(windows_by_climb) if limit_climbs is None else sorted(
            {m[0] for e in (v for s in recs_by_shard.values() for v in s.values())
             for mode in e.values() for m in mode} | set(limit_climbs))):
        holds_by_shard.setdefault(cid[:2].lower(), {})[cid] = [
            [h["x"], h["y"], h["role_id"]] for h in eng.get_climb_holds(cid)]
    for shard, payload in holds_by_shard.items():
        dump(os.path.join(out_dir, "holds", f"{shard}.json"), payload)

    # names.json: every indexed climb (or those reachable in a --limit run)
    if limit_climbs is None:
        name_ids = sorted(windows_by_climb)
    else:
        name_ids = sorted({c for s in holds_by_shard.values() for c in s})
    names = []
    for cid in name_ids:
        row = eng.db.execute(
            "SELECT name, setter_username, created_at FROM climbs WHERE uuid = ?", (cid,)).fetchone()
        names.append([cid, row[0], row[1], row[2], angles(cid)])
    names.sort(key=lambda r: (r[3] or ""))  # created_at order, as the live lookup sorts
    dump(os.path.join(out_dir, "names.json"), names)
    dump(os.path.join(out_dir, "meta.json"), {"top_k": TOP_K, "climbs": len(name_ids)})


def dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-repo", default=".", help="checkout containing server/db and retrieval.py")
    ap.add_argument("--out", default="data")
    ap.add_argument("--work", default="work")
    ap.add_argument("--limit", type=int, help="only export the first N climbs (smoke test)")
    ap.add_argument("--stage", choices=["search", "assemble", "all"], default="all")
    args = ap.parse_args()

    retrieval = load_engine_module(args.engine_repo)
    eng = Engine(retrieval, args.engine_repo)

    limit_climbs, work = None, args.work
    if args.limit:
        ids = sorted(r[0] for r in eng.db.execute("SELECT DISTINCT climb_id FROM subpattern_occurrences"))
        limit_climbs = ids[: args.limit]
        work = f"{args.work}_limit{args.limit}"

    if args.stage in ("search", "all"):
        stage_search(eng, work, limit_climbs)
    if args.stage in ("assemble", "all"):
        stage_assemble(eng, work, args.out, limit_climbs)


if __name__ == "__main__":
    main()
