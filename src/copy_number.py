"""
Copy number analysis functions for single-cell genomic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


def make_unif_dict(all_data, reassigned_populations, data_lengths):
    """Make uniformity dictionary for each population."""
    unif_dict = {}
    dev_from_median = {}
    for pop in reassigned_populations:
        pop_cells = reassigned_populations[pop]["cells"]
        pop_data = all_data[pop_cells]
        unif = pop_data.sum(axis=1)/data_lengths
        unif = unif/np.median(unif)
        unif_dict[pop] = unif
        dev_from_median[pop] = np.median(np.abs(unif - 1))
    min_dev_pop = min(dev_from_median, key=dev_from_median.get)
    return unif_dict, min_dev_pop

def compute_read_counts_relative_to_normal(all_data, populations, normal_pop_name):
    """ Average read counts in the cluster devided by the average read counts in the normal cluster"""
    mean_read_counts = {}
    for pop_name, pop_info in populations.items():
        cells = pop_info['cells']
        mean_count = all_data[cells].sum(axis=0).mean()
        mean_read_counts[pop_name] = mean_count
        #print(f"Mean read count for population {pop_name}: {mean_count:.2f}")
    normal_avg = mean_read_counts[normal_pop_name].copy()
    for pop_name, _ in populations.items():
        mean_read_counts[pop_name] = mean_read_counts[pop_name]/normal_avg
        print(f"Mean read count ratio for population {pop_name} relative to normal ({normal_pop_name}): {mean_read_counts[pop_name]:.2f}")
    
    return mean_read_counts


def compute_average_segment_counts(all_data, populations, data):
    """
    Compute average readcount per cell in each segment for each population,
    and ratios relative to the normal population.
    """
    unif_dict, normal_pop_name = make_unif_dict(all_data, populations, data["lengths"])
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

def compute_optimal_ploidy_and_profile(ratios_df, relative_counts, pop_col, m_range=[1.5, 6], step_size=0.1, 
                                     find_local_minima=True, window_size=2):
    """
    Compute optimal ploidy multiplier and integer copy number profile for a population.
    
    Parameters:
    - ratios_df: DataFrame with ratio data and segment information
    - relative_counts: Average read counts for the population relative to normal
    - pop_col: Column name for the population ratios (e.g., 'A1111_ratio')
    - m_range: Range of multipliers to test as [min, max] (default: [1.5, 6])
    - step_size: Step size for multiplier range (default: 0.1)
    - find_local_minima: Whether to find and return local minima (default: False)
    - window_size: Number of points to check on each side for local minima (default: 2)
    
    Returns:
    - optimal_m: Best multiplier value
    - min_cost: Minimum cost achieved
    - results_df: DataFrame with scaled and integer copy number profiles
    - cost_results: Dictionary of all tested multipliers and their costs
    - local_minima: List of tuples (multiplier, cost) for local minima (if find_local_minima=True)
    """
    
    # Create range of multipliers to test
    m_values = np.arange(m_range[0], m_range[1] + step_size, step_size)
    
    # Get ratios and segment lengths
    ratios = ratios_df[pop_col].values
    lengths = (ratios_df['end'] - ratios_df['start']).values
    
    cost_results = {}
    
    # Test each multiplier
    for m in m_values:
        # Compute integer copy numbers
        scaled_ratios = m * ratios
        integer_copies = np.round(scaled_ratios).astype(int)
        
        # Compute weighted cost: sum of length * (scaled_ratio - integer)^2
        deviations = scaled_ratios - integer_copies
        weighted_deviations_squared = lengths * (deviations ** 2)
        total_cost = np.sum(weighted_deviations_squared)
        
        cost_results[m] = total_cost
    
    # Find local minima if requested
    local_minima = []
    if find_local_minima:
        # Sort by multiplier value for local minima detection
        sorted_items = sorted(cost_results.items())
        m_vals = [item[0] for item in sorted_items]
        costs = [item[1] for item in sorted_items]
        
        # Check each point (excluding edges that don't have enough neighbors)
        for i in range(window_size, len(costs) - window_size):
            current_cost = costs[i]
            current_m = m_vals[i]
            
            # Check if current point is lower than all points in the window
            is_local_min = True
            
            # Check left side
            for j in range(i - window_size, i):
                if costs[j] <= current_cost:
                    is_local_min = False
                    break
            
            # Check right side
            if is_local_min:
                for j in range(i + 1, i + window_size + 1):
                    if costs[j] <= current_cost:
                        is_local_min = False
                        break
            
            if is_local_min:
                local_minima.append((current_m, current_cost))
        
        # Sort local minima by cost (best first)
        local_minima.sort(key=lambda x: x[1])

        # Find optimal multiplier:
        # 1) take the two local minima closest to target_m
        # 2) among those, pick the one with lowest cost_results[m]
        target_m = 2 * relative_counts[pop_col.replace('_ratio', '')]

        candidate_ms = [m for m, _ in local_minima]
        if len(candidate_ms) == 0:
            # Fallback if no local minima found
            candidate_ms = list(cost_results.keys())

        closest_two = sorted(candidate_ms, key=lambda m: abs(m - target_m))[:2]
        optimal_m = min(closest_two, key=lambda m: cost_results[m])
        
    else:
        # Find optimal multiplier
        optimal_m = min(cost_results.keys(), key=lambda x: cost_results[x])
    # Compute integer profile with optimal multiplier
    min_cost = cost_results[optimal_m]
    optimal_scaled_ratios = optimal_m * ratios
    integer_profile = np.round(optimal_scaled_ratios).astype(int)

    # Create results DataFrame
    results_df = ratios_df[['chrom', 'start', 'end']].copy()
    results_df[f'{pop_col}_scaled'] = optimal_scaled_ratios
    results_df[f'{pop_col}_integer'] = integer_profile
    
    # Print summary
    total_length = np.sum(lengths)
    avg_copy_number = np.sum(integer_profile * lengths) / total_length
    
    print(f"\nOptimal Results for {pop_col.replace('_ratio', '')}:")
    print(f"Optimal multiplier (m): {optimal_m:.3f}")
    print(f"Minimum cost: {min_cost:.2e}")
    print(f"Average copy number: {avg_copy_number:.2f}")
    
    if find_local_minima:
        print(f"Found {len(local_minima)} local minima")
    
    # Return results
    if find_local_minima:
        return optimal_m, min_cost, results_df, cost_results, local_minima
    else:
        return optimal_m, min_cost, results_df, cost_results
    
def create_copy_number_matrix_fast(all_data, reassigned_populations, ratios_to_normal, ploidies, normal_pop_name):
    """
    Optimized version of copy number matrix creation using vectorized operations.
    
    Parameters:
    - all_data: DataFrame with segments as rows and cells as columns
    - reassigned_populations: Dictionary of populations with their cell lists
    - ratios_to_normal: DataFrame with ratio data
    - ploidies: Dictionary with optimal multipliers for each population
    - normal_pop_name: Name of the normal population
    
    Returns:
    - heatmap_df: DataFrame with copy numbers (segments cells)
    - cell_pop_mapping: Dictionary mapping cells to their populations
    - pop_boundaries: List of tuples (pop_name, start_idx, end_idx)
    - segment_info: DataFrame with segment chromosome and position info
    """
    import numpy as np
    import pandas as pd
    
    # Create cell to population mapping
    cell_pop_mapping = {}
    for pop_name, pop_data in reassigned_populations.items():
        for cell in pop_data["cells"]:
            cell_pop_mapping[cell] = pop_name
    
    # Order cells by population
    ordered_cells = []
    pop_boundaries = []
    current_pos = 0
    
    # Add normal population first
    normal_cells = reassigned_populations[normal_pop_name]["cells"]
    ordered_cells.extend(normal_cells)
    current_pos += len(normal_cells)
    pop_boundaries.append((normal_pop_name, 0, current_pos))
    
    # Add other populations
    for pop_name in reassigned_populations.keys():
        if pop_name != normal_pop_name:
            pop_cells = reassigned_populations[pop_name]["cells"]
            ordered_cells.extend(pop_cells)
            start_pos = current_pos
            current_pos += len(pop_cells)
            pop_boundaries.append((pop_name, start_pos, current_pos))
    
    # Order segments by chromosome and position
    segment_order = ratios_to_normal[['chrom', 'start', 'end']].copy()
    chrom_order = ['chr' + str(i) for i in range(1, 23)] + ['chrX']
    segment_order['chrom_num'] = segment_order['chrom'].apply(
        lambda x: chrom_order.index(x) if x in chrom_order else 999
    )
    segment_order = segment_order.sort_values(['chrom_num', 'start']).reset_index(drop=True)
    ordered_segment_indices = segment_order.index
    
    # Get ordered data (this is the key optimization)
    ordered_data = all_data.iloc[ordered_segment_indices][ordered_cells]
    
    print(f"Processing {len(ordered_segment_indices)} segments using vectorized operations...")
    
    # Step 1: Compute normal_factor (segment average for normal cells)
    normal_cells_data = ordered_data.loc[:, ordered_data.columns.isin(normal_cells)]
    normal_factor = normal_cells_data.mean(axis=1).values  # Average across normal cells for each segment
    normal_factor = np.where(normal_factor > 0, normal_factor, 1)  # Avoid division by zero
    
    print("Normal factor computed")
    
    # Step 2: Normalize all cells by normal_factor
    normalized_data = ordered_data.div(normal_factor, axis=0)
    
    print("Data normalized by normal factor")
    
    # Step 3: For each cell, normalize by mean over all segments in that cell
    cell_means = normalized_data.mean(axis=0)  # Mean for each cell across all segments
    cell_means = np.where(cell_means > 0, cell_means, 1)  # Avoid division by zero
    
    # Normalize each cell by its own mean
    doubly_normalized_data = normalized_data.div(cell_means, axis=1)
    
    print("Data normalized by cell means")
    
    # Step 4: Apply ploidy multipliers
    heatmap_data = np.zeros_like(doubly_normalized_data.values, dtype=float)
    
    for i, cell in enumerate(ordered_cells):
        cell_pop = cell_pop_mapping[cell]
        
        # Get appropriate multiplier
        if cell_pop == normal_pop_name:
            multiplier = 2.0  # Normal cells get multiplier of 2
        else:
            pop_ratio_col = f"{cell_pop}_ratio"
            multiplier = ploidies.get(pop_ratio_col, 2.0)  # Use ploidy multiplier or default to 2
        
        # Apply multiplier to this cell's data
        heatmap_data[:, i] = doubly_normalized_data.iloc[:, i].values * multiplier
    
    print("Ploidy multipliers applied")
    
    # Create segment labels
    segment_labels = [
        f"{segment_order.loc[seg_idx, 'chrom']}:{segment_order.loc[seg_idx, 'start']}-{segment_order.loc[seg_idx, 'end']}"
        for seg_idx in ordered_segment_indices
    ]
    
    # Create heatmap DataFrame
    heatmap_df = pd.DataFrame(heatmap_data, columns=ordered_cells, index=segment_labels)
    
    # Create segment info DataFrame
    segment_info = segment_order[['chrom', 'start', 'end', 'chrom_num']].copy()
    segment_info.index = segment_labels
    
    print("Copy number matrix created successfully!")
    
    return heatmap_df, cell_pop_mapping, pop_boundaries, segment_info
