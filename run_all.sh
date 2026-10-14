#!/usr/bin/env bash
# Reproduces every result of the paper. Expect several hours on a 2-core machine.
set -euo pipefail
DATA=${DATA:-data}; LEARN=${LEARN:-data_learn}; RES=${RES:-results}; FIG=${FIG:-figures}; WORK=${WORK:-/tmp/gridtkg_bench}
mkdir -p "$DATA" "$LEARN" "$RES" "$FIG"
# 1. knowledge graphs (benchmark: 14 days at 15 min; learning: 1 year hourly)
for t in KG-S KG-M KG-L; do python3 grid_generator.py --tag $t --days 14 --step 15 --out "$DATA"; done
python3 grid_generator.py --tag KG-M --days 365 --step 60 --out "$LEARN"
# 2. system benchmark (run on an otherwise idle machine; PostgreSQL 16 binaries under /usr/lib/postgresql/16/bin)
for t in KG-S KG-M KG-L; do python3 bench_db.py --tag $t --data "$DATA" --out "$RES" --work "$WORK"; done
python3 bench_ablation.py --tag KG-L --data "$DATA" --out "$RES"
# 3. analytics layer
python3 learn_tasks.py --data "$LEARN/KG-M.npz" --out "$RES" --seeds 3
python3 ppo_maint.py --data "$LEARN/KG-M.npz" --res "$RES" --seeds 3
# 4. statistics and figures
python3 stats_db.py "$RES" "$DATA" "$FIG"
python3 make_figures_ml.py "$RES" "$FIG"
python3 make_diagrams.py "$FIG"
