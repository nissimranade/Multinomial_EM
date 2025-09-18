"""
EM clustering algorithms for single-cell genomic data.
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import chi2_contingency
from sklearn.linear_model import RANSACRegressor, LinearRegression
import matplotlib.pyplot as plt
from random import random


def single_random_seed(q_table):
    """Generate a single random seed for EM initialization."""
    s = []
    for j in range(len(q_table)):
        l2 = q_table.loc[j].values
        mi, mx = min(l2), max(l2)
        s.append([random() * (mx - mi) + mi, random() * (mx - mi) + mi])
    return s


def make_random_seeds(q_data, num_seeds=10):
    """Generate multiple random seeds for EM initialization."""
    Seeds = []
    for i in range(num_seeds):
        s = []
        for j in range(len(q_data)):
            l2 = q_data.loc[j].values
            mi, mx = min(l2), max(l2)
            s.append([random() * (mx - mi) + mi, random() * (mx - mi) + mi])
        Seeds.append(s)
    return Seeds


def EM_algorithm(data, s, MIN_DIFF=0.001):
    """Core EM algorithm implementation."""
    obs_NV = np.array(data)[:,:].T
    log_likelihood = -np.Inf
    
    mult_KV = np.array(s).T
    mult_KV = mult_KV / np.sum(mult_KV, axis=1)[:, np.newaxis]

    for i in range(20):
        logmult_KV = np.log(mult_KV)
        logprob_NK = np.dot(obs_NV, logmult_KV.T)
        logsum_N = [logsumexp(list(x)) for x in logprob_NK]
        logprob_NK = np.add(logprob_NK, -np.array(logsum_N)[:, np.newaxis])

        membership_NK = np.exp(logprob_NK)
        log_likelihood_last = log_likelihood
        log_likelihood = np.sum(logsum_N)

        mult_KV = np.dot(membership_NK.T, obs_NV)
        mult_KV = mult_KV / np.sum(mult_KV, axis=1)[:, np.newaxis]
        
        log_diff = abs(log_likelihood-log_likelihood_last)
        if log_diff < MIN_DIFF:
            break
            
    if log_diff > 10*MIN_DIFF:
        return logprob_NK, -np.Inf, mult_KV
    
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
    q_data = all_data/all_data.sum(axis=0)
    Seeds = make_random_seeds(q_data, num_seeds)

    best_log_lik = -np.Inf
    for s in Seeds:
        print("Trying new seed")
        logprob_NK, log_lik, mult_KV = EM_algorithm(all_data, s)
        if log_lik > best_log_lik:
            best_log_lik = log_lik
            best_logprob_NK = logprob_NK
            best_mult_KV = mult_KV
    
    return best_logprob_NK, best_log_lik, best_mult_KV


def ratio_of_ratios(all_data, threshold=4/5, name=""):
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


def recursive_em_split(data, cell_list, mult_KV_param=None, threshold=3/4, max_depth=10, current_depth=0, name="A"):
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
    
    if len(cell_list) < 50:
        print(f"{'  ' * current_depth}Population {name} too small to split")
        return {name: {'cells': cell_list, 'mult_KV': mult_KV_param}}
    
    try:
        current_data = data[cell_list]
        slopes, top_indices, bottom_indices, cells_1, cells_2, mult_KV = ratio_of_ratios(current_data, threshold, name)
        
        max_slope = max(slopes)
        min_slope = min(slopes)
        
        print(f"{'  ' * current_depth}Slopes for {name}: max={max_slope:.3f}, min={min_slope:.3f}")
        
        if threshold <= min_slope and max_slope <= 1/threshold:
            print(f"{'  ' * current_depth}Split rejected for {name}: slopes between {threshold:.3f} and {1/threshold:.3f}")
            return {name: {'cells': cell_list, 'mult_KV': mult_KV_param}}
        
        if len(cells_1) < 25 or len(cells_2) < 25:
            print(f"{'  ' * current_depth}Split produces too small subpopulations for {name}")
            return {name: {'cells': cell_list, 'mult_KV': mult_KV_param}}
        
        print(f"{'  ' * current_depth}Split accepted for {name}: {len(cells_1)} + {len(cells_2)} cells")
        
        result = {}
        
        sub_result_1 = recursive_em_split(
            data, cells_1, mult_KV[0], threshold, max_depth, current_depth + 1, name + "1"
        )
        result.update(sub_result_1)
        
        sub_result_2 = recursive_em_split(
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
