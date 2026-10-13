"""
Run recursive club splitting for a single tumor and save A/B/C run outputs.

Scope: ONLY the club-splitting pipeline from notebooks/EM club identities.ipynb
  load bincounts -> all_data -> recursive_club_split x3 (A, B, C) -> save tables

All three runs use: n_seeds=100, min_club_size=2, max_seed_difference=1,
threshold=3/4, max_depth=10.

Usage:
    python scripts/run_club_split.py --tumor t24
    for t in t24 t25 t31 t46 t57; do
        python scripts/run_club_split.py --tumor $t > logs/club_$t.log 2>&1 &
    done
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from random import random

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import chi2_contingency
from sklearn.linear_model import TheilSenRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fixed run settings (matching notebook runs A, B, C)
N_SEEDS = 100
MIN_CLUB_SIZE = 2
MAX_SEED_DIFFERENCE = 1
THRESHOLD = 3 / 4
MAX_DEPTH = 10


# ---------------------------------------------------------------------
# Club-splitting functions (copied 1:1 from the notebook)
# ---------------------------------------------------------------------
def make_random_seeds(q_data, num_seeds=10):
    Seeds = []
    for i in range(num_seeds):
        s = []
        for j in range(len(q_data)):
            l2 = q_data.loc[j].values
            mi, mx = min(l2), max(l2)
            s.append([random() * (mx - mi) + mi, random() * (mx - mi) + mi])
        Seeds.append(s)
    return Seeds


def EM_algorithm(data, s, Min_Diff=0.001, Max_Iter=20):
    obs_NV = np.array(data)[:, :].T
    log_likelihood = -np.inf

    mult_KV = np.array(s).T
    mult_KV = mult_KV / np.sum(mult_KV, axis=1)[:, np.newaxis]

    for i in range(Max_Iter):
        logmult_KV = np.log(mult_KV)
        logprob_NK = np.dot(obs_NV, logmult_KV.T)
        logsum_N = [logsumexp(list(x)) for x in logprob_NK]
        logprob_NK = np.add(logprob_NK, -np.array(logsum_N)[:, np.newaxis])

        membership_NK = np.exp(logprob_NK)
        log_likelihood_last = log_likelihood
        log_likelihood = np.sum(logsum_N)

        mult_KV = np.dot(membership_NK.T, obs_NV)
        mult_KV = mult_KV / np.sum(mult_KV, axis=1)[:, np.newaxis]

        log_diff = abs(log_likelihood - log_likelihood_last)
        if log_diff < Min_Diff:
            break

    if log_diff > 10 * Min_Diff:
        return logprob_NK, -np.inf, mult_KV

    return logprob_NK, log_likelihood, mult_KV


def split_is_informative(slopes, threshold):
    if len(slopes) == 0:
        return False
    lo = min(threshold, 1.0 / threshold)
    hi = 1.0 / lo
    min_s = min(slopes)
    max_s = max(slopes)
    return not (lo <= min_s and max_s <= hi)


def split_slopes_ratio_of_ratios(all_data, cells_1, cells_2, top_k=10, bottom_k=30):
    counts_1 = all_data[cells_1].sum(axis=1).to_numpy(dtype=float)
    counts_2 = all_data[cells_2].sum(axis=1).to_numpy(dtype=float)
    total_1 = counts_1.sum()
    total_2 = counts_2.sum()

    chi2_values = []
    for i in range(len(counts_1)):
        observed = np.array([
            [counts_1[i], total_1 - counts_1[i]],
            [counts_2[i], total_2 - counts_2[i]]
        ])
        chi2, _, _, _ = chi2_contingency(observed, correction=False)
        chi2_values.append(chi2)

    chi2_values = np.array(chi2_values)
    top_indices = np.argsort(chi2_values)[-top_k:][::-1]
    bottom_indices = np.argsort(chi2_values)[:bottom_k]

    slopes = []
    for i in top_indices:
        r1 = []
        r2 = []
        for j in bottom_indices:
            if j == i:
                continue
            if counts_1[j] <= 0 or counts_2[j] <= 0:
                continue
            r1.append(counts_1[i] / counts_1[j])
            r2.append(counts_2[i] / counts_2[j])

        if len(r1) >= 2:
            model = TheilSenRegressor(fit_intercept=False).fit(
                np.array(r1).reshape(-1, 1),
                np.array(r2)
            )
            slopes.append(float(model.coef_[0]))

    return slopes


def club_diameter(identity_array, members):
    # Largest Hamming distance between any two cells of the club
    rows = identity_array[members]
    diameter = 0
    for i in range(len(rows) - 1):
        diameter = max(diameter, int(np.sum(rows[i + 1:] != rows[i], axis=1).max()))
    return diameter


def find_clubs(
    data,
    n_seeds=100,
    min_club_size=25,
    max_seed_difference=0,
    threshold=4 / 5,
):
    cells = data.columns.tolist()

    q_data = data / data.sum(axis=0)
    Seeds = make_random_seeds(q_data, n_seeds)

    identities = []

    print("Starting EM club finding")
    for i, s in enumerate(Seeds):
        logprob_NK, log_lik, mult_KV = EM_algorithm(data, s)

        if not np.isfinite(log_lik):
            continue

        diffs = np.array([x - y for x, y in logprob_NK])
        cluster1 = np.where(diffs > 0)[0]
        cluster2 = np.where(diffs < 0)[0]

        cells_1 = [cells[j] for j in cluster1]
        cells_2 = [cells[j] for j in cluster2]

        # Reject weak splits: assign the same identity to every cell
        slopes = split_slopes_ratio_of_ratios(data, cells_1, cells_2)
        if not split_is_informative(slopes, threshold):
            assignment = np.zeros(len(cells), dtype=int)
        else:
            assignment = np.array([int(x > y) for x, y in logprob_NK])

        identities.append(assignment)

    print(f"Completed EM runs. Successful runs: {len(identities)}")
    if len(identities) == 0:
        print("No valid EM runs.")
        return [], cells, None, []

    identity_array = np.array(identities).T

    identity_df = pd.DataFrame(
        identity_array,
        index=cells,
        columns=[f"run_{i}" for i in range(len(identities))]
    )

    # Build clubs: connected components of the near-identity graph
    n_cells = len(cells)
    neighbors = [[] for _ in range(n_cells)]

    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            hamming_distance = np.sum(identity_array[i] != identity_array[j])
            if hamming_distance <= max_seed_difference:
                neighbors[i].append(j)
                neighbors[j].append(i)

    visited = np.zeros(n_cells, dtype=bool)
    all_clubs = []

    for start in range(n_cells):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component = []
        component_idx = []

        while stack:
            current = stack.pop()
            component.append(cells[current])
            component_idx.append(current)

            for neighbor in neighbors[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        all_clubs.append((component, club_diameter(identity_array, component_idx)))

    all_clubs.sort(key=lambda c: len(c[0]), reverse=True)

    major_clubs = [club for club, _ in all_clubs if len(club) >= min_club_size]
    major_diameters = [d for club, d in all_clubs if len(club) >= min_club_size]
    all_clubs = [club for club, _ in all_clubs]

    unassigned = [
        cell
        for club in all_clubs
        if len(club) < min_club_size
        for cell in club
    ]

    print("\nClub sizes:")
    print([len(club) for club in all_clubs])
    print(f"Major clubs: {[len(club) for club in major_clubs]}")
    print(f"Unassigned: {len(unassigned)}")

    return major_clubs, unassigned, identity_df, major_diameters


def recursive_club_split(
    data,
    n_seeds=100,
    min_club_size=25,
    max_seed_difference=1,
    max_depth=10,
    depth=0,
    name="root",
    threshold=3 / 4,
    diameter=None,
    diameters=None,
):
    if diameters is None:
        diameters = {}
    diameters[name] = diameter

    indent = "  " * depth
    n_cells = data.shape[1]

    print(
        f"{indent}{name}: "
        f"{n_cells} cells "
        f"(depth {depth})"
    )

    if n_cells < min_club_size:
        print(
            f"{indent}Population smaller than minimum club size. "
            f"Marking as unassigned."
        )
        return {}, {name: data.columns.tolist()}, diameters

    if depth >= max_depth:
        print(f"{indent}Maximum depth reached.")
        return {name: data.columns.tolist()}, {}, diameters

    if n_cells < 2 * min_club_size:
        print(
            f"{indent}Valid club, but too small to split "
            f"into two clubs of at least {min_club_size}. Stopping."
        )
        return {name: data.columns.tolist()}, {}, diameters

    major_clubs, unassigned, _, club_diameters = find_clubs(
        data,
        n_seeds=n_seeds,
        min_club_size=min_club_size,
        max_seed_difference=max_seed_difference,
        threshold=threshold
    )

    print(
        f"{indent}Major clubs: "
        f"{[len(club) for club in major_clubs]}"
    )
    print(f"{indent}Unassigned: {len(unassigned)}")

    if len(major_clubs) <= 1:
        print(f"{indent}No split. Stopping.")
        return {name: data.columns.tolist()}, {}, diameters

    print(
        f"{indent}Splitting into "
        f"{len(major_clubs)} clubs."
    )

    final_clusters = {}
    unassigned_dict = {}

    if len(unassigned) > 0:
        unassigned_dict[name] = unassigned

    for i, club in enumerate(major_clubs):
        child_name = f"{name}.{i}"
        child_data = data.loc[:, club]

        child_clusters, child_unassigned, _ = recursive_club_split(
            child_data,
            n_seeds=n_seeds,
            min_club_size=min_club_size,
            max_seed_difference=max_seed_difference,
            max_depth=max_depth,
            depth=depth + 1,
            name=child_name,
            threshold=threshold,
            diameter=club_diameters[i],
            diameters=diameters
        )

        final_clusters.update(child_clusters)
        unassigned_dict.update(child_unassigned)

    return final_clusters, unassigned_dict, diameters


# ---------------------------------------------------------------------
# Save helpers (same as the notebook save cell)
# ---------------------------------------------------------------------
def safe_write_table(df, path_no_ext):
    try:
        df.to_parquet(path_no_ext + ".parquet", index=False)
        return path_no_ext + ".parquet"
    except Exception:
        df.to_csv(path_no_ext + ".csv", index=False)
        return path_no_ext + ".csv"


def build_tables(clusters_dict, unassigned_dict, diameters=None):
    diameters = diameters or {}
    if not clusters_dict:
        tree_df = pd.DataFrame(columns=["id", "parent", "value", "depth", "is_leaf", "diameter"])
        leaf_df = pd.DataFrame(columns=["cluster_label", "n_cells", "depth", "parent_label"])
        assign_df = pd.DataFrame(columns=["cell_id", "leaf_cluster"])
        unassigned_df = pd.DataFrame(
            [{"node": node, "cell_id": c} for node, cells in unassigned_dict.items() for c in cells]
        ) if unassigned_dict else pd.DataFrame(columns=["node", "cell_id"])
        return tree_df, leaf_df, assign_df, unassigned_df

    leaf_rows = []
    assign_rows = []
    for cluster_label, cells in clusters_dict.items():
        depth = cluster_label.count(".")
        parent = "" if "." not in cluster_label else cluster_label.rsplit(".", 1)[0]
        leaf_rows.append({
            "cluster_label": cluster_label,
            "n_cells": len(cells),
            "depth": depth,
            "parent_label": parent
        })
        for c in cells:
            assign_rows.append({
                "cell_id": c,
                "leaf_cluster": cluster_label
            })

    leaf_df = pd.DataFrame(leaf_rows).sort_values("cluster_label").reset_index(drop=True)
    assign_df = pd.DataFrame(assign_rows).sort_values(["leaf_cluster", "cell_id"]).reset_index(drop=True)

    node_counts = {}
    for row in leaf_rows:
        label = row["cluster_label"]
        n_cells = row["n_cells"]
        parts = label.split(".")
        for i in range(1, len(parts) + 1):
            node = ".".join(parts[:i])
            node_counts[node] = node_counts.get(node, 0) + int(n_cells)

    leaf_set = set([r["cluster_label"] for r in leaf_rows])
    tree_rows = []
    for node, count in node_counts.items():
        tree_rows.append({
            "id": node,
            "parent": "" if "." not in node else node.rsplit(".", 1)[0],
            "value": int(count),
            "depth": node.count("."),
            "is_leaf": node in leaf_set,
            "diameter": diameters.get(node)
        })
    tree_df = pd.DataFrame(tree_rows)
    tree_df["diameter"] = tree_df["diameter"].astype("Int64")
    tree_df = tree_df.sort_values(["depth", "id"]).reset_index(drop=True)

    unassigned_rows = []
    for node, cells in unassigned_dict.items():
        for c in cells:
            unassigned_rows.append({
                "node": node,
                "cell_id": c
            })
    unassigned_df = pd.DataFrame(unassigned_rows) if unassigned_rows else pd.DataFrame(columns=["node", "cell_id"])

    return tree_df, leaf_df, assign_df, unassigned_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run recursive club split (A/B/C) for one tumor.")
    parser.add_argument("--tumor", required=True, help="Tumor id, e.g. t24")
    args = parser.parse_args()

    tumor = args.tumor
    data_dir = PROJECT_ROOT / "data" / tumor

    print(f"=== Tumor {tumor} ===")
    print(f"Data dir: {data_dir}")
    print(
        f"Settings: n_seeds={N_SEEDS}, min_club_size={MIN_CLUB_SIZE}, "
        f"max_seed_difference={MAX_SEED_DIFFERENCE}, threshold={THRESHOLD:.4f}, "
        f"max_depth={MAX_DEPTH}"
    )

    # ---- Data loading (same as notebook) ----
    bincounts = pd.read_csv(data_dir / f"{tumor}.bin_count_table_all_cells.txt", sep="\t")

    all_cell_names = [
        col for col in bincounts.columns
        if col not in ["chrom", "abstart", "abend", "lengths", "loc.start", "loc.end"]
    ]
    cells = pd.read_csv(
        data_dir / f"{tumor}.not_broken_final_list.txt",
        header=None
    )[0].values

    data = bincounts[["chrom", "abstart", "abend"] + all_cell_names].copy()
    data["lengths"] = data["abend"] - data["abstart"]
    all_data = data[cells].copy()

    print(f"Loaded {all_data.shape[1]} cells x {all_data.shape[0]} segments")

    # ---- Three independent recursive club-split runs ----
    run_kwargs = dict(
        n_seeds=N_SEEDS,
        min_club_size=MIN_CLUB_SIZE,
        max_seed_difference=MAX_SEED_DIFFERENCE,
        threshold=THRESHOLD,
        max_depth=MAX_DEPTH,
    )

    final_clusters_A, unassigned_A, diameters_A = recursive_club_split(all_data, name="A", **run_kwargs)
    final_clusters_B, unassigned_B, diameters_B = recursive_club_split(all_data, name="B", **run_kwargs)
    final_clusters_C, unassigned_C, diameters_C = recursive_club_split(all_data, name="C", **run_kwargs)

    # ---- Save (same layout as notebook save cell) ----
    run_map = {
        "A": (final_clusters_A, unassigned_A, diameters_A),
        "B": (final_clusters_B, unassigned_B, diameters_B),
        "C": (final_clusters_C, unassigned_C, diameters_C),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "data" / tumor / "results_bundle" / f"triple_run_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    written_files = []
    for label, (clusters_dict, unassigned_dict, diameters) in run_map.items():
        tree_df, leaf_df, assign_df, unassigned_df = build_tables(clusters_dict, unassigned_dict, diameters)
        tag = f"{label}{MAX_SEED_DIFFERENCE}"

        written_files.append(safe_write_table(tree_df, os.path.join(out_dir, f"{tumor}_{tag}_cluster_tree")))
        written_files.append(safe_write_table(leaf_df, os.path.join(out_dir, f"{tumor}_{tag}_leaf_clusters")))
        written_files.append(safe_write_table(assign_df, os.path.join(out_dir, f"{tumor}_{tag}_cell_assignments")))
        written_files.append(safe_write_table(unassigned_df, os.path.join(out_dir, f"{tumor}_{tag}_unassigned_cells")))

        summary.append({
            "label": tag,
            "n_leaf_clusters": int(len(leaf_df)),
            "n_tree_nodes": int(len(tree_df)),
            "n_assigned_cells": int(assign_df["cell_id"].nunique()) if len(assign_df) else 0,
            "n_unassigned_cells": int(unassigned_df["cell_id"].nunique()) if len(unassigned_df) else 0
        })

    meta = {
        "tumor": tumor,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "params": {
            "n_seeds": N_SEEDS,
            "min_club_size": MIN_CLUB_SIZE,
            "max_seed_difference": MAX_SEED_DIFFERENCE,
            "threshold": THRESHOLD,
            "max_depth": MAX_DEPTH,
        },
        "runs": summary,
        "files": written_files
    }
    with open(os.path.join(out_dir, f"{tumor}_run_manifest{MAX_SEED_DIFFERENCE}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved all 3 runs to: {out_dir}")
    print(pd.DataFrame(summary))


if __name__ == "__main__":
    main()
