"""
Visualization functions for single-cell copy number analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


def plot_copy_number_step_function(ratios_df, pop_cols, title="Copy Number Comparison"):
    """
    Plot copy number ratios as step functions with chromosome boundaries.
    All populations are superimposed on the same plot.
    
    Parameters:
    - ratios_df: DataFrame with ratio data (ratios_to_normal_1)
    - pop_cols: List of population column names (e.g., ['A1111_ratio', 'A1112_ratio'])
    - title: Plot title
    """
    
    # Sort data by chromosome and position
    plot_data = ratios_df.copy()
    
    # Define chromosome order
    chrom_order = ['chr' + str(i) for i in range(1, 23)] + ['chrX']
    plot_data['chrom_num'] = plot_data['chrom'].apply(
        lambda x: chrom_order.index(x) if x in chrom_order else 999
    )
    plot_data = plot_data.sort_values(['chrom_num', 'start']).reset_index(drop=True)
    
    # Calculate segment lengths and cumulative positions
    plot_data['length'] = plot_data['end'] - plot_data['start']
    plot_data['cum_start'] = 0
    plot_data['cum_end'] = 0
    
    cum_pos = 0
    chrom_boundaries = []
    chrom_centers = []
    chrom_labels = []
    
    for chrom in chrom_order:
        if chrom in plot_data['chrom'].values:
            chrom_data = plot_data[plot_data['chrom'] == chrom]
            
            # Set cumulative positions for this chromosome
            chrom_start = cum_pos
            for idx in chrom_data.index:
                plot_data.loc[idx, 'cum_start'] = cum_pos
                cum_pos += plot_data.loc[idx, 'length']
                plot_data.loc[idx, 'cum_end'] = cum_pos
            
            # Record chromosome boundaries and centers
            chrom_end = cum_pos
            chrom_boundaries.append(chrom_end)
            chrom_centers.append((chrom_start + chrom_end) / 2)
            chrom_labels.append(chrom.replace('chr', ''))
            
            # Add small gap between chromosomes
            cum_pos += max(plot_data['length']) * 0.02  # 2% of max segment length
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    
    # Define colors for different populations
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    # Plot each population on the same axes
    for pop_idx, pop_col in enumerate(pop_cols):
        color = colors[pop_idx % len(colors)]
        pop_label = pop_col.replace("_ratio", "")
        
        # Plot step function for this population
        for idx, row in plot_data.iterrows():
            x_coords = [row['cum_start'], row['cum_end']]
            y_coords = [row[pop_col], row[pop_col]]
            
            # Add label only to first segment of each population
            label = pop_label if idx == 0 else None
            ax.plot(x_coords, y_coords, color=color, linewidth=2, label=label)
            
            # Connect segments with vertical lines within same chromosome
            if idx > 0:
                prev_row = plot_data.iloc[idx-1]
                if row['chrom'] == prev_row['chrom']:  # Same chromosome
                    ax.plot([prev_row['cum_end'], row['cum_start']], 
                           [prev_row[pop_col], row[pop_col]], color=color, linewidth=1)
    
    # Add chromosome boundary lines
    for boundary in chrom_boundaries[:-1]:  # Don't add line after last chromosome
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7)
    
    # Add reference lines for copy numbers
    ax.axhline(y=1, color='black', linestyle='-', alpha=0.5, label='Normal (2 copies)')
    ax.axhline(y=1.5, color='lightgray', linestyle=':', alpha=0.5)
    ax.axhline(y=0.5, color='lightgray', linestyle=':', alpha=0.5)
    ax.axhline(y=2, color='lightgray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='red', linestyle=':', alpha=0.5)

    # Set title and labels
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Copy Number Ratio', fontsize=14)
    ax.set_xlabel('Chromosome', fontsize=14)
    
    # Set y-axis limits based on all populations
    all_ratios = []
    for pop_col in pop_cols:
        all_ratios.extend(plot_data[pop_col].values)
    
    y_max = max(all_ratios) * 1.1
    y_min = max(0, min(all_ratios) * 0.9)  # Don't go below 0
    ax.set_ylim(y_min, y_max)
    
    # Set chromosome labels on x-axis
    ax.set_xticks(chrom_centers)
    ax.set_xticklabels(chrom_labels, rotation=0)
    
    # Add legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


def plot_copy_number_heatmap(heatmap_df, pop_boundaries, segment_info, title="Copy Number Heatmap"):
    """
    Plot the copy number heatmap with segments displayed proportional to their genomic size.
    
    Parameters:
    - heatmap_df: DataFrame with copy numbers (segments × cells)
    - pop_boundaries: List of tuples (pop_name, start_idx, end_idx)
    - segment_info: DataFrame with segment chromosome info
    - title: Plot title
    """
    
    # Calculate segment lengths
    segment_lengths = segment_info['end'] - segment_info['start']
    
    # Normalize segment lengths to determine how many rows each segment gets
    # Scale so that smallest segment gets 1 row, largest gets proportional amount
    min_length = segment_lengths.min()
    max_rows_per_segment = 10  # Maximum rows any single segment can occupy
    
    # Calculate rows per segment based on length (minimum 1 row)
    rows_per_segment = np.maximum(1, np.round(segment_lengths / min_length).astype(int))
    # Cap at maximum to prevent huge chromosomes from dominating
    rows_per_segment = np.minimum(rows_per_segment, max_rows_per_segment)
    
    # Create expanded heatmap data
    expanded_data = []
    expanded_chroms = []
    chrom_positions = {}  # Track position of each chromosome
    current_row = 0
    
    print(f"Expanding heatmap from {len(heatmap_df)} segments to {rows_per_segment.sum()} rows...")
    
    for i, (seg_label, row_data) in enumerate(heatmap_df.iterrows()):
        chrom = segment_info.iloc[i]['chrom']
        repeats = rows_per_segment.iloc[i]
        
        # Track chromosome boundaries
        if chrom not in chrom_positions:
            chrom_positions[chrom] = {'start': current_row, 'rows': 0}
        
        # Add this segment's data multiple times based on its size
        for _ in range(repeats):
            expanded_data.append(row_data.values)
            expanded_chroms.append(chrom)
            current_row += 1
            chrom_positions[chrom]['rows'] += 1
    
    expanded_data = np.array(expanded_data)
    
    # Calculate chromosome centers for labeling
    chrom_centers = []
    chrom_labels = []
    chrom_order = ['chr' + str(i) for i in range(1, 23)] + ['chrX']
    
    for chrom in chrom_order:
        if chrom in chrom_positions:
            start = chrom_positions[chrom]['start']
            rows = chrom_positions[chrom]['rows']
            center = start + rows / 2
            chrom_centers.append(center)
            chrom_labels.append(chrom)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 15))
    
    # Create custom colormap: blue (deletions) -> white (CN=2) -> red (amplifications)
    colors_list = [
        (0.0, 'darkblue'),     # CN = 0
        (0.25, 'blue'),        # CN = 1
        (0.5, 'white'),        # CN = 2 (normal)
        (0.75, 'red'),         # CN = 3-4
        (1.0, 'darkred')       # CN = 5+
    ]
    
    cmap = LinearSegmentedColormap.from_list('copy_number', colors_list, N=256)
    norm = TwoSlopeNorm(vmin=0, vcenter=2, vmax=6)
    
    # Plot heatmap
    im = ax.imshow(expanded_data, aspect='auto', cmap=cmap, norm=norm)
    
    # Add population boundaries (vertical lines only - no text labels here)
    for pop_name, start, end in pop_boundaries:
        ax.axvline(x=start-0.5, color='black', linewidth=2)
        ax.axvline(x=end-0.5, color='black', linewidth=2)
    
    # Add chromosome boundaries (horizontal lines)
    current_pos = 0
    for chrom in chrom_order:
        if chrom in chrom_positions:
            current_pos += chrom_positions[chrom]['rows']
            if chrom != chrom_order[-1]:  # Don't add line after last chromosome
                ax.axhline(y=current_pos-0.5, color='gray', linewidth=1, alpha=0.7)
    
    # Set labels and title
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Cells (ordered by population)', fontsize=12)
    
    # Set population tick labels at bottom (this is the only place we add labels)
    cell_tick_positions = [boundary[1] + (boundary[2] - boundary[1]) / 2 for boundary in pop_boundaries]
    cell_tick_labels = [boundary[0] for boundary in pop_boundaries]
    ax.set_xticks(cell_tick_positions)
    ax.set_xticklabels(cell_tick_labels, rotation=45)
    
    # Set chromosome labels on y-axis (one label per chromosome at center)
    ax.set_yticks(chrom_centers)
    ax.set_yticklabels(chrom_labels)
    
    # Add colorbar with simple numeric labels
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Copy Number', fontsize=12)
    
    # Set colorbar ticks to just numbers
    cbar_ticks = [0, 1, 2, 3, 4, 5, 6]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(['0', '1', '2', '3', '4', '5', '6+'])
    
    # Print chromosome size information
    print("\nChromosome representation in heatmap:")
    for chrom in chrom_order:
        if chrom in chrom_positions:
            rows = chrom_positions[chrom]['rows']
            # Calculate total genomic length for this chromosome
            chrom_segs = segment_info[segment_info['chrom'] == chrom]
            total_length = (chrom_segs['end'] - chrom_segs['start']).sum()
            print(f"{chrom}: {rows} rows, {total_length:,} bp total")
    
    plt.tight_layout()
    plt.show()


def print_overlap_matrix(pop1, pop2, name1="A", name2="B"):
    """
    Print overlap matrix showing number of shared cells between populations
    """
    pop1_names = sorted(pop1.keys())
    pop2_names = sorted(pop2.keys())
    
    # Print header
    print(f"\nOverlap Matrix ({name1} rows vs {name2} cols):")
    print("=" * 50)
    print(f"{'Pop':<8}", end="")
    for pop_b in pop2_names:
        print(f"{pop_b:>8}", end="")
    print()
    
    # Print overlap counts
    for pop_a in pop1_names:
        cells_a = set(pop1[pop_a]["cells"])
        print(f"{pop_a:<8}", end="")
        
        for pop_b in pop2_names:
            cells_b = set(pop2[pop_b]["cells"])
            overlap = len(cells_a.intersection(cells_b))
            print(f"{overlap:>8}", end="")
        print()
