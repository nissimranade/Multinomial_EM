"""
Copy number analysis functions for single-cell genomic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


def make_unif_dict(all_data, reassigned_populations, data):
    """Make uniformity dictionary for each population."""
    unif_dict = {}
    dev_from_median = {}
    for pop in reassigned_populations:
        pop_cells = reassigned_populations[pop]["cells"]
        pop_data = all_data[pop_cells]
        unif = pop_data.sum(axis=1)/data["lengths"]
        unif = unif/np.median(unif)
        unif_dict[pop] = unif
        dev_from_median[pop] = np.median(np.abs(unif - 1))
    min_dev_pop = min(dev_from_median, key=dev_from_median.get)
    return unif_dict, min_dev_pop


def compute_average_segment_counts(all_data, populations, data):
    """
    Compute average readcount per cell in each segment for each population,
    and ratios relative to the normal population.
    """
    unif_dict, normal_pop_name = make_unif_dict(all_data, populations, data)
    print(f"Normal population identified as: {normal_pop_name}")
    
    avg_seg_count = pd.DataFrame()
    avg_seg_count['chrom'] = data['chrom'].values
    avg_seg_count['start'] = data['abstart'].values  
    avg_seg_count['end'] = data['abend'].values
    avg_seg_count['segment_id'] = range(len(data))
    
    for pop_name, pop_data in populations.items():
        cells_in_pop = pop_data["cells"]
        
        if len(cells_in_pop) > 0:
            pop_segment_data = all_data[cells_in_pop]
            avg_counts = pop_segment_data.sum(axis=1) / len(cells_in_pop)
            avg_seg_count[pop_name] = avg_counts
        else:
            avg_seg_count[pop_name] = 0
    
    ratios_to_normal = pd.DataFrame()
    ratios_to_normal['chrom'] = avg_seg_count['chrom']
    ratios_to_normal['start'] = avg_seg_count['start']
    ratios_to_normal['end'] = avg_seg_count['end']
    ratios_to_normal['segment_id'] = avg_seg_count['segment_id']
    
    normal_avg_counts = avg_seg_count[normal_pop_name]
    
    for pop_col, _ in populations.items():
        if pop_col == normal_pop_name:
            ratios_to_normal[f"{pop_col}_ratio"] = 1.0
        else:
            epsilon = 1e-10
            safe_normal_counts = normal_avg_counts + epsilon
            ratios_to_normal[f"{pop_col}_ratio"] = avg_seg_count[pop_col] / safe_normal_counts
    
    return avg_seg_count, ratios_to_normal, normal_pop_name


def compute_optimal_ploidy_and_profile(ratios_df, pop_col, m_range=[1.5, 6], step_size=0.1, plot=True):
    """
    Compute optimal ploidy multiplier and integer copy number profile for a population.
    """
    m_values = np.arange(m_range[0], m_range[1] + step_size, step_size)
    ratios = ratios_df[pop_col].values
    lengths = (ratios_df['end'] - ratios_df['start']).values
    
    cost_results = {}
    
    for m in m_values:
        scaled_ratios = m * ratios
        integer_copies = np.round(scaled_ratios).astype(int)
        deviations = scaled_ratios - integer_copies
        weighted_deviations_squared = lengths * (deviations ** 2)
        total_cost = np.sum(weighted_deviations_squared)
        cost_results[m] = total_cost
    
    optimal_m = min(cost_results.keys(), key=lambda x: cost_results[x])
    min_cost = cost_results[optimal_m]
    
    optimal_scaled_ratios = optimal_m * ratios
    integer_profile = np.round(optimal_scaled_ratios).astype(int)
    
    results_df = ratios_df[['chrom', 'start', 'end']].copy()
    results_df[f'{pop_col}_scaled'] = optimal_scaled_ratios
    results_df[f'{pop_col}_integer'] = integer_profile
    
    if plot:
        _plot_ploidy_optimization(ratios_df, pop_col, cost_results, optimal_m, results_df)
    
    return optimal_m, min_cost, results_df, cost_results


def _plot_ploidy_optimization(ratios_df, pop_col, cost_results, optimal_m, results_df):
    """Helper function to plot ploidy optimization results."""
    # Implementation of the plotting logic from your notebook
    # (This would be the plotting code from compute_optimal_ploidy_and_profile)
    pass


def create_copy_number_matrix_fast(all_data, reassigned_populations, ratios_to_normal, ploidies, normal_pop_name):
    """
    Optimized version of copy number matrix creation using vectorized operations.
    """
    cell_pop_mapping = {}
    for pop_name, pop_data in reassigned_populations.items():
        for cell in pop_data["cells"]:
            cell_pop_mapping[cell] = pop_name
    
    ordered_cells = []
    pop_boundaries = []
    current_pos = 0
    
    normal_cells = reassigned_populations[normal_pop_name]["cells"]
    ordered_cells.extend(normal_cells)
    current_pos += len(normal_cells)
    pop_boundaries.append((normal_pop_name, 0, current_pos))
    
    for pop_name in sorted(reassigned_populations.keys()):
        if pop_name != normal_pop_name:
            pop_cells = reassigned_populations[pop_name]["cells"]
            ordered_cells.extend(pop_cells)
            start_pos = current_pos
            current_pos += len(pop_cells)
            pop_boundaries.append((pop_name, start_pos, current_pos))
    
    segment_order = ratios_to_normal[['chrom', 'start', 'end']].copy()
    chrom_order = ['chr' + str(i) for i in range(1, 23)] + ['chrX']
    segment_order['chrom_num'] = segment_order['chrom'].apply(
        lambda x: chrom_order.index(x) if x in chrom_order else 999
    )
    segment_order = segment_order.sort_values(['chrom_num', 'start']).reset_index(drop=True)
    ordered_segment_indices = segment_order.index
    
    ordered_data = all_data.iloc[ordered_segment_indices][ordered_cells]
    
    print(f"Processing {len(ordered_segment_indices)} segments using vectorized operations...")
    
    normal_cells_data = ordered_data.loc[:, ordered_data.columns.isin(normal_cells)]
    normal_factor = normal_cells_data.mean(axis=1).values
    normal_factor = np.where(normal_factor > 0, normal_factor, 1)
    
    normalized_data = ordered_data.div(normal_factor, axis=0)
    cell_means = normalized_data.mean(axis=0)
    cell_means = np.where(cell_means > 0, cell_means, 1)
    doubly_normalized_data = normalized_data.div(cell_means, axis=1)
    
    heatmap_data = np.zeros_like(doubly_normalized_data.values, dtype=float)
    
    for i, cell in enumerate(ordered_cells):
        cell_pop = cell_pop_mapping[cell]
        
        if cell_pop == normal_pop_name:
            multiplier = 2.0
        else:
            pop_ratio_col = f"{cell_pop}_ratio"
            multiplier = ploidies.get(pop_ratio_col, 2.0)
        
        heatmap_data[:, i] = doubly_normalized_data.iloc[:, i].values * multiplier
    
    segment_labels = [
        f"{segment_order.loc[seg_idx, 'chrom']}:{segment_order.loc[seg_idx, 'start']}-{segment_order.loc[seg_idx, 'end']}"
        for seg_idx in ordered_segment_indices
    ]
    
    heatmap_df = pd.DataFrame(heatmap_data, columns=ordered_cells, index=segment_labels)
    
    segment_info = segment_order[['chrom', 'start', 'end', 'chrom_num']].copy()
    segment_info.index = segment_labels
    
    print("Copy number matrix created successfully!")
    
    return heatmap_df, cell_pop_mapping, pop_boundaries, segment_info
