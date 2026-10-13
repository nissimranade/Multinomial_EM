"""
EM clustering algorithms for single-cell genomic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import chi2_contingency
from sklearn.linear_model import RANSACRegressor, LinearRegression
from random import random


def single_random_seed(q_table):
    """Generate a single random seed for EM initialization."""
    s = []
    for j in range(len(q_table)):
        l2 = q_table.loc[j].values
        mi, mx = min(l2), max(l2)
        s.append([random() * (mx - mi) + mi, random() * (mx - mi) + mi])
    return s


def make_random_seeds(q_data, num_seeds=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    seeds = []
    for _ in range(num_seeds):
        s = []
        for j in range(len(q_data)):
            vals = q_data.loc[j].values
            mi, mx = np.min(vals), np.max(vals)
            s.append([rng.uniform(mi, mx), rng.uniform(mi, mx)])
        seeds.append(s)
    return seeds

def EM_algorithm(data, s, Min_Diff=0.001, Max_Iter=20):
    obs_NV = np.array(data).T
    log_likelihood = -np.inf

    mult_KV = np.array(s).T
    mult_KV = mult_KV / np.sum(mult_KV, axis=1)[:, np.newaxis]

    log_diff = np.inf
    for _ in range(Max_Iter):
        logmult_KV = np.log(mult_KV)
        logprob_NK = np.dot(obs_NV, logmult_KV.T)
        logsum_N = np.array([logsumexp(x) for x in logprob_NK])
        logprob_NK = logprob_NK - logsum_N[:, np.newaxis]

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


def alg_on_single_seed(all_data):
    """Run EM algorithm on a single random seed."""
    q_data = all_data/all_data.sum(axis=0)
    s = single_random_seed(q_data)
    print("single random seed done")
    logprob_NK, log_lik, mult_KV = EM_algorithm(all_data, s)
    return logprob_NK, log_lik, mult_KV


def alg_on_best_of_seeds(all_data, num_seeds=10):
    """Run EM algorithm on multiple seeds and return best result."""
    q_data = all_data / all_data.sum(axis=0)
    seeds = make_random_seeds(q_data, num_seeds=num_seeds)

    best_log_lik = -np.inf
    best_logprob_NK, best_mult_KV = None, None
    for s in seeds:
        print("Trying new seed")
        logprob_NK, log_lik, mult_KV = EM_algorithm(all_data, s)
        if log_lik > best_log_lik:
            best_log_lik = log_lik
            best_logprob_NK = logprob_NK
            best_mult_KV = mult_KV

    return best_logprob_NK, best_log_lik, best_mult_KV

def calculate_overlap(cells_a, cells_b):
    """Calculate Jaccard similarity between two cell lists."""
    set_a = set(cells_a)
    set_b = set(cells_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0

def consensus_split_population(all_data, consensus_threshold=0.9, min_consensus=0.3, num_seeds=50, rng=None):
    """
    Select split by descending log-likelihood, but require minimum consensus support.
    A candidate split is accepted if at least `min_consensus` fraction of OTHER splits
    have overlap >= `consensus_threshold` (flip-aware).
    """
    q_data = all_data / all_data.sum(axis=0)
    Seeds = make_random_seeds(q_data, num_seeds, rng=rng)

    split_records = []
    for s in Seeds:
        logprob_NK, log_lik, mult_KV = EM_algorithm(all_data, s)

        # Skip non-converged / invalid runs
        if not np.isfinite(log_lik):
            continue

        diffs = np.array([x - y for x, y in logprob_NK])
        cluster1 = np.where(diffs > 0)[0]
        cluster2 = np.where(diffs < 0)[0]
        unassigned = np.where(diffs == 0)[0]

        cells_1 = [all_data.columns[i] for i in cluster1]
        cells_2 = [all_data.columns[i] for i in cluster2]
        unassigned_cells = [all_data.columns[i] for i in unassigned]

        # Keep only proper binary splits
        if len(cells_1) == 0 or len(cells_2) == 0:
            continue

        split_records.append({
            "log_lik": log_lik,
            "cells_1": cells_1,
            "cells_2": cells_2,
            "unassigned": unassigned_cells,
            "mult_KV": mult_KV
        })

    if len(split_records) == 0:
        print("No valid EM splits found.")
        return None, None, list(all_data.columns)

    # Sort by log-likelihood (highest first)
    split_records.sort(key=lambda r: r["log_lik"], reverse=True)

    # If only one valid split exists, accept it directly
    if len(split_records) == 1:
        chosen = split_records[0]
        print("Only one valid split found; accepting highest log-likelihood split.")
        print(f"Cluster 1: {len(chosen['cells_1'])} cells")
        print(f"Cluster 2: {len(chosen['cells_2'])} cells")
        print(f"Unassigned (stragglers): {len(chosen['unassigned'])} cells")
        return chosen["cells_1"], chosen["cells_2"], chosen["unassigned"], chosen["mult_KV"]

    # Evaluate candidates in descending log-likelihood order
    n = len(split_records)
    for i, cand in enumerate(split_records):
        supporters = 0

        for j, other in enumerate(split_records):
            if i == j:
                continue

            c1_i, c2_i = cand["cells_1"], cand["cells_2"]
            c1_j, c2_j = other["cells_1"], other["cells_2"]

            overlap_11 = calculate_overlap(c1_i, c1_j)
            overlap_22 = calculate_overlap(c2_i, c2_j)
            overlap_12 = calculate_overlap(c1_i, c2_j)
            overlap_21 = calculate_overlap(c2_i, c1_j)

            overlap_score = max(
                (overlap_11 + overlap_22) / 2,
                (overlap_12 + overlap_21) / 2
            )

            if overlap_score >= consensus_threshold:
                supporters += 1

        support_fraction = supporters / (n - 1)

        if support_fraction >= min_consensus:
            print(
                f"Selected split rank={i+1} by log-likelihood "
                f"(log_lik={cand['log_lik']:.4f}) with support={support_fraction:.3f}"
            )
            print(f"Cluster 1: {len(cand['cells_1'])} cells")
            print(f"Cluster 2: {len(cand['cells_2'])} cells")
            print(f"Unassigned (stragglers): {len(cand['unassigned'])} cells")
            return cand["cells_1"], cand["cells_2"], cand["unassigned"], cand["mult_KV"]

    print("No split met minimum consensus support in log-likelihood order.")
    return None, None, list(all_data.columns), None


def ratio_of_ratios(all_data, consensus_threshold = 0.9, min_consensus = 0.3, num_seeds = 50,threshold=4/5, name="", plot=True, rng=None):
    """Ratio of ratios analysis for population splitting."""
    cells_1, cells_2, unassigned, mult_KV = consensus_split_population(all_data, consensus_threshold=consensus_threshold, min_consensus=min_consensus, num_seeds=num_seeds, rng=rng)

    if cells_1 is None or cells_2 is None:
        print("Consensus split failed.")
        return [], [], [], [], [], all_data.columns.tolist(), None
    
    counts_1 = all_data[cells_1].sum(axis=1)
    counts_2 = all_data[cells_2].sum(axis=1)
    
    total_1 = np.sum(counts_1)
    total_2 = np.sum(counts_2)

    chi2_values = []
    for i in range(len(counts_1)):
        observed = np.array([
            [counts_1[i], total_1 - counts_1[i]],
            [counts_2[i], total_2 - counts_2[i]]
        ])
        chi2, _, _, _ = chi2_contingency(observed, correction=False)
        chi2_values.append(chi2)
    chi2_values = np.array(chi2_values)
    top_indices = np.argsort(chi2_values)[-10:][::-1]
    bottom_indices = np.argsort(chi2_values)[:30]
    
    ratios_1 = []
    ratios_2 = []
    for i in top_indices:
        r1 = []
        r2 = []
        for j in bottom_indices:
            if j!=i:
                r1.append(counts_1[i]/counts_1[j])
                r2.append(counts_2[i]/counts_2[j])
        ratios_1.append(r1)
        ratios_2.append(r2)
    slopes = []
    if plot:
        fig,ax = plt.subplots(1, figsize=(10, 10))

    for r1, r2 in zip(ratios_1, ratios_2):
        if plot:
            plt.scatter(r1, r2, alpha=0.5)
    for r1, r2 in zip(ratios_1, ratios_2):
        model = RANSACRegressor(LinearRegression(fit_intercept=False)).fit(np.array(r1).reshape(-1, 1), np.array(r2))
        slopes.append(model.estimator_.coef_[0])
        if plot:
            x_fit = np.linspace(min(r1), max(r1), 100).reshape(-1, 1)
            y_fit = model.predict(x_fit)
            plt.plot(x_fit, y_fit, linestyle='dashed')

    if plot:
        x = np.linspace(0, 10, 50)
        plt.plot(x, x, color='black')  # y=x line

        y = threshold * x
        plt.plot(x, y, color='grey')  # y=(4/5)x line
        y = (1/threshold) * x
        plt.plot(x, y, color='grey')  # y=(5/4)x line

        l1 = str(len(cells_1))
        l2 = str(len(cells_2))
        plt.title(f"Ratio of Ratios Plot {name}1({l1}) and {name}2({l2})")
        plt.xlim([0,10])
        plt.ylim([0,10])
    return slopes, top_indices, bottom_indices, cells_1, cells_2, unassigned, mult_KV


def ratio_of_ratios_legacy(all_data, threshold=4/5, name=""):
    """Ratio of ratios analysis for population splitting."""
    logprob_NK, log_lik, mult_KV = alg_on_best_of_seeds(all_data)
    cluster1 = np.where(np.array([x-y for x,y in logprob_NK])>0)[0]
    cluster2 = np.where(np.array([x-y for x,y in logprob_NK])<0)[0]
    cells_1 = [all_data.columns[i] for i in cluster1]
    cells_2 = [all_data.columns[i] for i in cluster2]
    
    counts_1 = all_data[cells_1].sum(axis=1)
    counts_2 = all_data[cells_2].sum(axis=1)
    
    total_1 = np.sum(counts_1)
    total_2 = np.sum(counts_2)

    chi2_values = []
    for i in range(len(counts_1)):
        observed = np.array([
            [counts_1[i], total_1 - counts_1[i]],
            [counts_2[i], total_2 - counts_2[i]]
        ])
        chi2, p, dof, expected = chi2_contingency(observed, correction=False)
        chi2_values.append(chi2)
    chi2_values = np.array(chi2_values)
    top_indices = np.argsort(chi2_values)[-10:][::-1]
    bottom_indices = np.argsort(chi2_values)[:30]
    
    ratios_1 = []
    ratios_2 = []
    for i in top_indices:
        r1 = []
        r2 = []
        for j in bottom_indices:
            if j!=i:
                r1.append(counts_1[i]/counts_1[j])
                r2.append(counts_2[i]/counts_2[j])
        ratios_1.append(r1)
        ratios_2.append(r2)
    slopes = []
    fig,ax = plt.subplots(1, figsize=(10, 10))
    
    for r1, r2 in zip(ratios_1, ratios_2): 
        plt.scatter(r1, r2, alpha=0.5)
    for r1, r2 in zip(ratios_1, ratios_2): 
        model = RANSACRegressor(LinearRegression(fit_intercept=False)).fit(np.array(r1).reshape(-1, 1), np.array(r2))
        slopes.append(model.estimator_.coef_[0])
        x_fit = np.linspace(min(r1), max(r1), 100).reshape(-1, 1)
        y_fit = model.predict(x_fit)
        plt.plot(x_fit, y_fit, linestyle='dashed')

    x = np.linspace(0, 10, 50)
    plt.plot(x, x, color='black')  # y=x line 

    y = threshold * x
    plt.plot(x, y, color='grey')  # y=(4/5)x line
    y = (1/threshold) * x
    plt.plot(x, y, color='grey')  # y=(5/4)x line

    l1 = str(len(cells_1))
    l2 = str(len(cells_2))
    plt.title(f"Ratio of Ratios Plot {name}1({l1}) and {name}2({l2})")
    plt.xlim([0,10])
    plt.ylim([0,10])
    return slopes, top_indices, bottom_indices, cells_1, cells_2, mult_KV

def recursive_em_split(data, cell_list, mult_KV_param=None,
    consensus_threshold=0.8,
    min_consensus=0.2,
    num_seeds=50,
    threshold=3/4,
    max_depth=10,
    current_depth=0,
    min_cluster_size = 25,
    name="A",
    plot=True,
    rng=None):
    """
    Recursively splits cell populations using EM algorithm until slopes are within threshold.
    """
    print(f"{'  ' * current_depth}Processing population {name} with {len(cell_list)} cells")
    
    if mult_KV_param is None:
        current_data = data[cell_list]
        q_data = current_data / current_data.sum(axis=0)
        mult_KV_param = q_data.mean(axis=1).values
    
    threshold = min(threshold, 1/threshold)
    if current_depth >= max_depth:
        print(f"{'  ' * current_depth}Max depth reached for {name}")
        return {name: {'cells': cell_list, 'mult_KV': mult_KV_param}}
    
    if len(cell_list) < 2*min_cluster_size:
        print(f"{'  ' * current_depth}Population {name} too small to split")
        return {name: {'cells': cell_list, 'mult_KV': mult_KV_param}}
    
    try:
        current_data = data[cell_list]
        slopes, _, _, cells_1, cells_2, _, mult_KV = ratio_of_ratios(current_data, consensus_threshold=consensus_threshold, min_consensus=min_consensus, num_seeds=num_seeds, threshold=threshold, name=name, plot=plot, rng=rng)
        if not slopes:
            print(f"{'  ' * current_depth}Consensus split failed for {name}")
            return {name: {'cells': cell_list, 'mult_KV': mult_KV_param}}
        
        max_slope = max(slopes)
        min_slope = min(slopes)
        
        print(f"{'  ' * current_depth}Slopes for {name}: max={max_slope:.3f}, min={min_slope:.3f}")
        
        if threshold <= min_slope and max_slope <= 1/threshold:
            print(f"{'  ' * current_depth}Split rejected for {name}: slopes between {threshold:.3f} and {1/threshold:.3f}")
            return {name: {'cells': cell_list, 'mult_KV': mult_KV_param}}
        
        if len(cells_1) < min_cluster_size or len(cells_2) < min_cluster_size:
            print(f"{'  ' * current_depth}Split produces too small subpopulations for {name}")
            return {name: {'cells': cell_list, 'mult_KV': mult_KV_param}}
        
        print(f"{'  ' * current_depth}Split accepted for {name}: {len(cells_1)} + {len(cells_2)} cells")
        
        result = {}
        
        sub_result_1 = recursive_em_split(
            data, cells_1, mult_KV_param=mult_KV[0], consensus_threshold=consensus_threshold, min_consensus=min_consensus, num_seeds=num_seeds, threshold=threshold, max_depth=max_depth, current_depth=current_depth + 1, name=name + "1", plot=plot, rng=rng
            )
        result.update(sub_result_1)
        
        sub_result_2 = recursive_em_split(
            data, cells_2, mult_KV_param=mult_KV[1], consensus_threshold=consensus_threshold, min_consensus=min_consensus, num_seeds=num_seeds, threshold=threshold, max_depth=max_depth, current_depth=current_depth + 1, name=name + "2", plot=plot, rng=rng
            )
        result.update(sub_result_2)
        
        return result
        
    except Exception as e:
        print(f"{'  ' * current_depth}Error splitting {name}: {e}")
        return {name: {'cells': cell_list, 'mult_KV': mult_KV_param}}


def recursive_em_split_legacy(data, cell_list, mult_KV_param=None,
    consensus_threshold=0.8,
    min_consensus=0.2,
    num_seeds=50,
    threshold=3/4,
    max_depth=10,
    current_depth=0,
    min_cluster_size = 25, 
    name="A"):

    """
    Recursively splits cell populations using EM algorithm until slopes are within threshold.
    """
    print(f"{'  ' * current_depth}Processing population {name} with {len(cell_list)} cells")
    
    if mult_KV_param is None:
        current_data = data[cell_list]
        q_data = current_data / current_data.sum(axis=0)
        mult_KV_param = q_data.mean(axis=1).values
    
    threshold = min(threshold, 1/threshold)
    if current_depth >= max_depth:
        print(f"{'  ' * current_depth}Max depth reached for {name}")
        return {name: {'cells': cell_list, 'mult_KV': mult_KV_param}}
    
    if len(cell_list) < 2*min_cluster_size:
        print(f"{'  ' * current_depth}Population {name} too small to split")
        return {name: {'cells': cell_list, 'mult_KV': mult_KV_param}}
    
    try:
        current_data = data[cell_list]
        slopes, top_indices, bottom_indices, cells_1, cells_2, unassigned, mult_KV = ratio_of_ratios(current_data, threshold, name)
        
        max_slope = max(slopes)
        min_slope = min(slopes)
        
        print(f"{'  ' * current_depth}Slopes for {name}: max={max_slope:.3f}, min={min_slope:.3f}")
        
        if threshold <= min_slope and max_slope <= 1/threshold:
            print(f"{'  ' * current_depth}Split rejected for {name}: slopes between {threshold:.3f} and {1/threshold:.3f}")
            return {name: {'cells': cell_list, 'mult_KV': mult_KV_param}}
        
        if len(cells_1) < min_cluster_size or len(cells_2) < min_cluster_size:
            print(f"{'  ' * current_depth}Split produces too small subpopulations for {name}")
            return {name: {'cells': cell_list, 'mult_KV': mult_KV_param}}
        
        print(f"{'  ' * current_depth}Split accepted for {name}: {len(cells_1)} + {len(cells_2)} cells")
        
        result = {}
        
        sub_result_1 = recursive_em_split_legacy(
            data, cells_1, mult_KV[0], threshold, max_depth, current_depth + 1, name + "1"
        )
        result.update(sub_result_1)
        
        sub_result_2 = recursive_em_split_legacy(
            data, cells_2, mult_KV[1], threshold, max_depth, current_depth + 1, name + "2"
        )
        result.update(sub_result_2)
        
        return result
        
    except Exception as e:
        print(f"{'  ' * current_depth}Error splitting {name}: {e}")
        return {name: {'cells': cell_list, 'mult_KV': mult_KV_param}}


def reassign_cells(data, populations):
    """Reassigns cells to populations based on closest multinomial parameters."""
    from scipy.special import logsumexp
    
    pop_params = {pop: populations[pop]['mult_KV'] for pop in populations}
    new_populations = {pop: {"cells": [], "mult_KV": populations[pop]["mult_KV"]} for pop in populations}

    for cell in data.columns:
        cell_data = data[cell].values
        log_likelihoods = {}
        for pop, params in pop_params.items():
            log_probs = np.log(params + 1e-10)
            log_likelihood = np.sum(cell_data * log_probs)
            log_likelihoods[pop] = log_likelihood
        
        max_log_likelihood = max(log_likelihoods.values())
        for pop in log_likelihoods:
            log_likelihoods[pop] -= max_log_likelihood
        
        best_pop = max(log_likelihoods, key=log_likelihoods.get)
        new_populations[best_pop]["cells"].append(cell)

    return new_populations
