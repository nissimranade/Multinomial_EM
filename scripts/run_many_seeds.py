"""
Run recursive EM splitting N times with independent random seeds and save
per-run cluster assignments to a CSV, checkpointed after every run.

Per-run pipeline (matches notebooks/EM many random seeds.ipynb):
    final_populations_A = recursive_em_split(all_data, cells, num_seeds=1,
        threshold=3/4, max_depth=10, name="A", plot=False, rng=rng)
    A_dict = reassign_cells(all_data, final_populations_A)

Output (wide format): one row per cell, one column per run (run_000, ...),
each entry the cluster name the cell was assigned to in that run.
Both output CSVs are rewritten after every run, so a crash at run k keeps
the results of runs 0..k-1. A failed run is recorded in the summary CSV
and simply has no column in the assignments CSV.

The per-run split trace (recursive_em_split's stdout) is redirected to
logs/em_many_seeds_<tumor>.log; the terminal only gets one line per run.

Each run uses a truly random seed drawn from OS entropy (secrets.randbits)
for the random module, numpy's global RNG (used by RANSAC), and a dedicated
np.random.default_rng(seed) threaded into make_random_seeds. The seed used
is recorded in the summary CSV and the log, so any single run can still be
reproduced after the fact if needed.

Usage:
    python scripts/run_many_seeds.py --tumor t57
    python scripts/run_many_seeds.py --tumor t57 --n-runs 500

    # All five tumors in parallel (one background process each):
    bash scripts/run_many_seeds_all.sh
"""

import argparse
import contextlib
import random
import secrets
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from em_clustering import recursive_em_split, reassign_cells

# Fixed run settings (matching the notebook)
N_RUNS = 500
NUM_SEEDS = 1
THRESHOLD = 3 / 4
MAX_DEPTH = 10
NAME = "A"


def load_data(tumor):
    """Replicates the data loading in notebooks/EM many random seeds.ipynb."""
    data_dir = PROJECT_ROOT / "data" / tumor
    bincounts = pd.read_csv(data_dir / f"{tumor}.bin_count_table_all_cells.txt", sep="\t")
    all_cell_names = [col for col in bincounts.columns if col not in
                      ['chrom', 'abstart', 'abend', 'lengths', 'loc.start', 'loc.end']]
    cells = pd.read_csv(data_dir / f"{tumor}.not_broken_final_list.txt", header=None)[0].values
    data = bincounts[["chrom", "abstart", "abend"] + all_cell_names].copy()
    data["lengths"] = data["abend"] - data["abstart"]
    all_data = data[cells].copy()
    return all_data, cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tumor", default="t57")
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--out-suffix", default="",
                        help="Suffix appended to the default output filename, e.g. "
                             "'_500runs' for a 500-run batch")
    parser.add_argument("--out", default=None,
                        help="Output CSV path "
                             "(default: data/<tumor>/<tumor>_em_many_seeds<suffix>_assignments.csv)")
    args = parser.parse_args()

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (PROJECT_ROOT / "data" / args.tumor /
                    f"{args.tumor}_em_many_seeds{args.out_suffix}_assignments.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.csv")
    log_path = PROJECT_ROOT / "logs" / f"em_many_seeds_{args.tumor}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    all_data, cells = load_data(args.tumor)
    print(f"Loaded {all_data.shape[0]} bins x {all_data.shape[1]} cells for {args.tumor}")

    print(f"Assignments -> {out_path}")
    print(f"Summary     -> {summary_path}")
    print(f"Split trace -> {log_path}")

    assignments = {}   # run label -> {cell: cluster}
    summary_rows = []

    with open(log_path, "a") as log_f:
        for run in range(args.n_runs):
            seed = secrets.randbits(32)   # truly random, from OS entropy
            label = f"run_{run:03d}"
            random.seed(seed)
            np.random.seed(seed)
            rng = np.random.default_rng(seed)

            try:
                with contextlib.redirect_stdout(log_f):
                    log_f.write(f"\n===== {label} (seed={seed}) =====\n")
                    final_populations = recursive_em_split(
                        all_data, cells, num_seeds=args.num_seeds,
                        threshold=THRESHOLD, max_depth=MAX_DEPTH, name=args.name,
                        plot=False, rng=rng,
                    )
                    assignment = reassign_cells(all_data, final_populations)

                assignments[label] = {
                    cell: pop
                    for pop, d in assignment.items()
                    for cell in d["cells"]
                }
                sizes = {pop: len(d["cells"]) for pop, d in assignment.items()}
                summary_rows.append({"run": label, "seed": seed,
                                     "n_clusters": len(assignment),
                                     "cluster_sizes": str(sizes)})
                print(f"{label}: {len(assignment)} clusters -> {sizes}")
            except Exception as e:
                summary_rows.append({"run": label, "seed": seed,
                                     "n_clusters": -1,
                                     "cluster_sizes": f"ERROR: {e}"})
                print(f"{label}: FAILED ({e})")

            # Checkpoint: rewrite outputs after every run
            pd.DataFrame(assignments).to_csv(out_path, index_label="cell")
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Done. {len(assignments)}/{args.n_runs} runs completed.")


if __name__ == "__main__":
    main()
