# GridTKG — temporal knowledge graph for unified OLTP/OLAP on power-grid data

This repository contains all the code behind the manuscript *Knowledge Graph Integration for Real-Time and Analytical Power Industry Applications*:

- the storage engine,
- the data generator,
- the baseline adapters,
- the benchmark harness,
- the analytics layer (GTN and PPO),
- the figure scripts,
- the raw result files.

All files sit in one folder.

## Files

| File | Purpose |
|---|---|
| `ontology.py` | CIM-aligned classes and predicates, constraint checks, Datalog rules, RDF snapshot export |
| `gridtkg_store.py` | GridTKG engine: CSR fact partitions with delta, slot-aligned measurement chunks, WAL with group commit, the 14 operations, rule materialisation |
| `grid_generator.py` | Builds KG-S / KG-M / KG-L from IEEE 118, PEGASE 1354 and PEGASE 9241 (pandapower) with DC power flow, LODF outages, degradation, storms, failures, maintenance and sensor artefacts |
| `baselines.py` | PostgreSQL 16, DuckDB and Kùzu+DuckDB implementations of the same operations (SQL and Cypher texts) |
| `bench_db.py` | Benchmark harness: validation, warm-up, 30 runs, mixed workload |
| `bench_ablation.py` | Design ablation (no materialised busNeighbor; row-log measurements) |
| `stats_db.py` | Confidence intervals, Welch tests with Holm correction, dataset statistics, Figs. 4–6 |
| `gtn_models.py` | GTN, GCN, GRU and MLP models |
| `learn_tasks.py` | Outage detection and 7-day failure prediction, with features read through the GridTKG API |
| `ppo_maint.py` | Maintenance environment, Plackett–Luce PPO, reference policies |
| `make_figures_ml.py` | Figs. 7–9 |
| `make_diagrams.py` | Figs. 1–3, drawn from scratch with matplotlib |
| `run_all.sh` | Reproduces everything |
| `make_numbers.py`, `make_numbers_ppo.py` | Collect every number quoted in the paper from the result files into `numbers.json` |
| `db_KG-S.json`, `db_KG-M.json`, `db_KG-L.json` | Raw per-call latencies of all systems (30 runs), validation answers, mixed workload |
| `ablation_KG-L.json`, `stats_db.json` | Design ablation; derived statistics |
| `learn_results.json`, `ppo_results.json` | Learning-task metrics per seed; maintenance-policy episodes and PPO curves |

## Requirements

- Python ≥ 3.11 and the packages in `requirements.txt` (CPU only).
- PostgreSQL 16 server binaries, for the PostgreSQL baseline. The harness runs `initdb` and `pg_ctl` from `/usr/lib/postgresql/16/bin`.

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh            # data, benchmark, ablation, learning, PPO, statistics, figures
```

The benchmark should run on an otherwise idle machine. The paper's timings were measured on 2 vCPU of a 2.1 GHz Xeon with 7 GB RAM. All randomness is seeded:

- **Data generator:** seed 7.
- **Benchmark runs:** seeds 1000–1029.
- **Learning:** model seeds 0–2.
- **PPO evaluation:** seeds 900000–900029.

## Minimal use of the engine

```python
from gridtkg_store import GridTKG, load_dataset, materialise
d = load_dataset("data/KG-M.npz")
s = GridTKG("wal.log"); s.bulk_load(d, materialise(d)); s.finalize_indexes(d["meta_blocks"]["Branch"])
bus = d["meta_blocks"]["Bus"][0]
print(s.khop(bus, 3, t=100))            # buses within 3 energised hops and their load
print(s.shortest_hops(bus, bus + 50, t=100))
```

## Licence

MIT. See `LICENSE`.
