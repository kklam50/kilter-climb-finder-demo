# Kilter Climb Finder — precomputed static demo

**This is a precomputed demo, not the live engine.** Every result you see is
stored output from an offline export of the retrieval engine in
[kilter-climb-finder](https://github.com/kklam50/kilter-climb-finder), served
as static JSON. The live engine (embedding model, memory-mapped vector index,
per-window search and aggregation) lives in that repo.

Given a Kilter Board climb, it shows other climbs with similar (or opposite)
movement patterns. Limitations versus the live app: only the 31,121 indexed
climbs, exact (case-insensitive) name matching only (no fuzzy fallback), and a
fixed top 5 per climb.

## How it works

1. `scripts/export_static_demo.py` reproduces `RetrievalEngine.recommend_for_climb`
   in batch form (blocked matrix multiplies, tie-faithful top-k, identical
   aggregation) for every indexed climb, both modes, `top_k=5`, and writes
   sharded JSON to `data/`.
2. `scripts/verify_static_demo.py` checks the export against the real engine
   (same climbs, order, scores within 1e-4, same window counts).
3. `scripts/build_static_demo.py` assembles `client/` + `data/` into
   `static_demo/`; `scripts/deploy_static_demo.sh` pushes it to a Hugging Face
   static Space.

`server/generate_training_data_v2/retrieval.py` is the engine, kept for the
export scripts; `server/db/` (gitignored) holds the engine's data.

## Rebuilding

```
pip install -r requirements.txt
python scripts/export_static_demo.py      # resumable; --limit N for a smoke test
python scripts/verify_static_demo.py
python scripts/build_static_demo.py
cd static_demo && python -m http.server   # try it locally
```

The demo is frozen at export time; re-run after any index or model change.
