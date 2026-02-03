"""
Visualization functions for single-cell copy number analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import seaborn as sns


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



def print_overlap_matrix(pop1, pop2, name1="A", name2="B"):
    """
    Print overlap matrix showing number of shared cells between populations
    and return it as a DataFrame
    
    Parameters:
    - pop1, pop2: population dictionaries
    - name1, name2: names for the population sets
    
    Returns:
    - overlap_df: DataFrame with overlap counts
    """
    pop1_names = sorted(pop1.keys())
    pop2_names = sorted(pop2.keys())
    
    # Create overlap matrix
    overlap_matrix = np.zeros((len(pop1_names), len(pop2_names)), dtype=int)
    
    # Fill overlap matrix
    for i, pop_a in enumerate(pop1_names):
        cells_a = set(pop1[pop_a]["cells"])
        for j, pop_b in enumerate(pop2_names):
            cells_b = set(pop2[pop_b]["cells"])
            overlap = len(cells_a.intersection(cells_b))
            overlap_matrix[i, j] = overlap
    
    # Create DataFrame
    overlap_df = pd.DataFrame(overlap_matrix, 
                             index=pop1_names, 
                             columns=pop2_names)
    
    # Print header
    print(f"\nOverlap Matrix ({name1} rows vs {name2} cols):")
    print("=" * 50)
    print(f"{'Pop':<8}", end="")
    for pop_b in pop2_names:
        print(f"{pop_b:>8}", end="")
    print()
    
    # Print overlap counts
    for pop_a in pop1_names:
        print(f"{pop_a:<8}", end="")
        for pop_b in pop2_names:
            overlap = overlap_df.loc[pop_a, pop_b]
            print(f"{overlap:>8}", end="")
        print()
    
    return overlap_df

def plot_overlap_matrix_figure(pop1, pop2, name1="A", name2="B", title=None, figsize=(10, 8)):
    """
    Create a colorful overlap matrix figure suitable for presentations
    
    Parameters:
    - pop1, pop2: population dictionaries
    - name1, name2: names for the population sets
    - title: custom title (if None, uses default)
    - figsize: figure size tuple
    """
    

    pop1_names = pop1.keys()
    pop2_names = pop2.keys()

    overlap_df = print_overlap_matrix(pop1, pop2, name1, name2)
    new_order = []
    for col in overlap_df.columns:
        #Find the index of the maximum value in the column
        max_index = overlap_df[col].idxmax()
        if max_index not in new_order:
            new_order.append(max_index)

    for row in overlap_df.index:
        if row not in new_order:
            new_order.append(row)

    overlap_df = overlap_df.reindex(index=new_order)

    pop1_names = overlap_df.index
    pop2_names = overlap_df.columns
    pop1_sizes = overlap_df.sum(axis=1).tolist()
    pop2_sizes = overlap_df.sum(axis=0).tolist()

    # Create overlap matrix as values from overlap_df
    overlap_matrix = overlap_df.values
    
    
    # Create custom colormap for better visualization
    # White for 0-15 cells, then gradually darker blue, capping at 50+
    colors = ['white', 'lightblue', 'steelblue', 'navy', 'darkblue']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom_blues', colors, N=n_bins)
    
    # Create normalized matrix for color mapping
    # Values 0-15 map to very light colors
    # Values 15-50 map to increasingly dark colors  
    # Values 50+ all map to darkest color
    color_matrix = np.copy(overlap_matrix)
    color_matrix = np.where(color_matrix > 50, 50, color_matrix)  # Cap at 50
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the main heatmap with custom color scaling
    im = ax.imshow(color_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=50)
    
    # Add text annotations with cell counts
    for i in range(len(pop1_names)):
        for j in range(len(pop2_names)):
            text = int(overlap_matrix[i, j])
            
            # Determine text color based on background
            if overlap_matrix[i, j] < 15:
                text_color = 'black'  # Dark text on light background
            elif overlap_matrix[i, j] < 30:
                text_color = 'white'  # White text on medium background
            else:
                text_color = 'white'  # White text on dark background
                
            ax.text(j, i, text, ha='center', va='center', 
                    fontsize=12, fontweight='bold', color=text_color)
    
    # Customize main heatmap
    ax.set_xticks(range(len(pop2_names)))
    ax.set_yticks(range(len(pop1_names)))
    ax.set_xticklabels(pop2_names, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(pop1_names, fontsize=12)
    ax.set_xlabel(f'{name2} Populations', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{name1} Populations', fontsize=14, fontweight='bold')
    
    # Add gridlines
    ax.set_xticks(np.arange(len(pop2_names)+1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(pop1_names)+1)-0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add colorbar with custom labels
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Number of Shared Cells', fontsize=12, fontweight='bold')
    
    # Set custom colorbar ticks
    cbar_ticks = [0, 15, 25, 35, 50]
    cbar_labels = ['0', '15', '25', '35', '50+']
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_labels)
    
    # Main title
    if title is None:
        title = f'Cell Overlap Matrix: {name1} vs {name2} Populations'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # Print the matrix in console as well
    total_cells_1 = sum(pop1_sizes)  # Calculate total cells in pop1
    total_cells_2 = sum(pop2_sizes)  # Calculate total cells in pop2
    
    print(f"\nOverlap Matrix ({name1} rows vs {name2} cols):")
    print("=" * 60)
    print(f"{'Pop':<8}", end="")
    for pop_b in pop2_names:
        print(f"{pop_b:>8}", end="")
    print(f"{'Total':>8}")
    print("-" * 60)
    
    for i, pop_a in enumerate(pop1_names):
        print(f"{pop_a:<8}", end="")
        for j, pop_b in enumerate(pop2_names):
            overlap = int(overlap_matrix[i, j])
            print(f"{overlap:>8}", end="")
        print(f"{pop1_sizes[i]:>8}")
    
    print("-" * 60)
    print(f"{'Total':<8}", end="")
    for j, size in enumerate(pop2_sizes):
        print(f"{size:>8}", end="")
    print(f"{total_cells_1:>8}")
    
    return overlap_df

def plot_cost_function_with_minima(cost_results, local_minima, pop_name, optimal_m):
    """
    Plot cost function with vertical lines at local minima.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot cost function
    m_values = list(cost_results.keys())
    costs = list(cost_results.values())
    
    ax.plot(m_values, costs, 'b-', linewidth=2, label='Cost function')
    
    # Mark global minimum
    ax.axvline(x=optimal_m, color='red', linestyle='-', linewidth=2, 
               label=f'Global minimum: m = {optimal_m:.2f}')
    
    # Mark all local minima
    colors = ['orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (m, cost) in enumerate(local_minima):
        if m != optimal_m:  # Don't duplicate the global minimum line
            color = colors[i % len(colors)]
            ax.axvline(x=m, color=color, linestyle='--', alpha=0.7,
                      label=f'Local min #{i+1}: m = {m:.2f}')
    
    # Mark points on the curve
    for i, (m, cost) in enumerate(local_minima):
        color = 'red' if m == optimal_m else colors[i % len(colors)]
        marker = 'o' if m == optimal_m else 's'
        size = 100 if m == optimal_m else 60
        
        ax.scatter([m], [cost], color=color, s=size, marker=marker, 
                  zorder=10, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Multiplier (m)', fontsize=12)
    ax.set_ylabel('Weighted Cost', fontsize=12)
    ax.set_title(f'Cost Function with Local Minima: {pop_name}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nLocal Minima Summary for {pop_name}:")
    print("-" * 50)
    for i, (m, cost) in enumerate(local_minima):
        status = " (GLOBAL)" if m == optimal_m else ""
        print(f"#{i+1}: m = {m:.2f}, cost = {cost:.2e}{status}")


def plot_ploidy_optimization(results_df, cost_results, optimal_m, pop_col):
    """
    Plot ploidy optimization results: cost function and copy number profile.
    
    Parameters:
    - results_df: DataFrame with copy number results
    - cost_results: Dictionary of multiplier costs
    - optimal_m: Optimal multiplier value
    - pop_col: Population column name
    """
    # Sort data by chromosome and position for plotting
    plot_data = results_df.copy()
    
    # Define chromosome order
    chrom_order = ['chr' + str(i) for i in range(1, 23)] + ['chrX']
    plot_data['chrom_num'] = plot_data['chrom'].apply(
        lambda x: chrom_order.index(x) if x in chrom_order else 999
    )
    plot_data = plot_data.sort_values(['chrom_num', 'start']).reset_index(drop=True)
    
    # Calculate cumulative positions
    plot_data['length'] = plot_data['end'] - plot_data['start']
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
            cum_pos += max(plot_data['length']) * 0.02
    
    # Get the final position (end of chrX)
    final_pos = plot_data['cum_end'].max()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    
    # Plot cost function
    m_plot_values = list(cost_results.keys())
    costs = list(cost_results.values())
    
    ax1.plot(m_plot_values, costs, 'b-', linewidth=2)
    ax1.axvline(x=optimal_m, color='red', linestyle='--', 
               label=f'Optimal m = {optimal_m:.2f}')
    ax1.set_xlabel('Multiplier (m)')
    ax1.set_ylabel('Weighted Cost')
    ax1.set_title(f'Ploidy Optimization for {pop_col.replace("_ratio", "")}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot step functions for scaled and integer copy numbers
    scaled_col = f'{pop_col}_scaled'
    integer_col = f'{pop_col}_integer'
    
    # Plot scaled copy numbers
    for idx, row in plot_data.iterrows():
        x_coords = [row['cum_start'], row['cum_end']]
        y_coords = [row[scaled_col], row[scaled_col]]
        
        # Add label only to first segment
        label = 'Scaled Copy Number' if idx == 0 else None
        ax2.plot(x_coords, y_coords, color='blue', linewidth=2, alpha=0.7, label=label)
        
        # Connect segments with vertical lines within same chromosome
        if idx > 0:
            prev_row = plot_data.iloc[idx-1]
            if row['chrom'] == prev_row['chrom']:
                ax2.plot([prev_row['cum_end'], row['cum_start']], 
                       [prev_row[scaled_col], row[scaled_col]], color='blue', linewidth=1, alpha=0.7)
    
    # Plot integer copy numbers
    for idx, row in plot_data.iterrows():
        x_coords = [row['cum_start'], row['cum_end']]
        y_coords = [row[integer_col], row[integer_col]]
        
        # Add label only to first segment
        label = 'Integer Copy Number' if idx == 0 else None
        ax2.plot(x_coords, y_coords, color='red', linewidth=2, alpha=0.8, label=label)
        
        # Connect segments with vertical lines within same chromosome
        if idx > 0:
            prev_row = plot_data.iloc[idx-1]
            if row['chrom'] == prev_row['chrom']:
                ax2.plot([prev_row['cum_end'], row['cum_start']], 
                       [prev_row[integer_col], row[integer_col]], color='red', linewidth=1, alpha=0.8)
    
    # Add chromosome boundary lines
    for boundary in chrom_boundaries[:-1]:
        ax2.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    # Add reference lines for copy numbers
    ax2.axhline(y=2, color='black', linestyle='-', alpha=0.5, label='Normal (2 copies)')
    ax2.axhline(y=1, color='lightgray', linestyle=':', alpha=0.5)
    ax2.axhline(y=3, color='lightgray', linestyle=':', alpha=0.5)
    ax2.axhline(y=4, color='lightgray', linestyle=':', alpha=0.5)
    
    # Set labels and formatting
    ax2.set_title(f'Copy Number Profile: {pop_col.replace("_ratio", "")}\nOptimal m = {optimal_m:.2f}')
    ax2.set_ylabel('Copy Number', fontsize=12)
    ax2.set_xlabel('Chromosome', fontsize=12)
    
    # Set chromosome labels on x-axis
    ax2.set_xticks(chrom_centers)
    ax2.set_xticklabels(chrom_labels, rotation=0)
    ax2.set_xlim(0,final_pos)
    
    # Set y-axis limits
    all_values = list(plot_data[scaled_col]) + list(plot_data[integer_col])
    y_max = max(all_values) * 1.1
    y_min = max(0, min(all_values) * 0.9)
    ax2.set_ylim(y_min, y_max)
    
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_ploidy_comparison(results_df, cost_results, optimal_m, local_m, pop_col, hap_count_df=None, cells=None):
    """
    Plot ploidy comparison between two multipliers with dual y-axes.
    
    Parameters:
    - results_df: DataFrame with copy number results (using optimal_m)
    - cost_results: Dictionary of multiplier costs
    - optimal_m: Optimal (global minimum) multiplier value
    - local_m: Local minimum multiplier to compare against
    - pop_col: Population column name
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Sort data by chromosome and position for plotting
    plot_data = results_df.copy()
    
    # Define chromosome order and calculate cumulative positions
    chrom_order = ['chr' + str(i) for i in range(1, 23)] + ['chrX']
    plot_data['chrom_num'] = plot_data['chrom'].apply(
        lambda x: chrom_order.index(x) if x in chrom_order else 999
    )
    plot_data = plot_data.sort_values(['chrom_num', 'start']).reset_index(drop=True)
    
    plot_data['length'] = plot_data['end'] - plot_data['start']
    cum_pos = 0
    chrom_boundaries = []
    chrom_centers = []
    chrom_labels = []
    
    for chrom in chrom_order:
        if chrom in plot_data['chrom'].values:
            chrom_data = plot_data[plot_data['chrom'] == chrom]
            chrom_start = cum_pos
            for idx in chrom_data.index:
                plot_data.loc[idx, 'cum_start'] = cum_pos
                cum_pos += plot_data.loc[idx, 'length']
                plot_data.loc[idx, 'cum_end'] = cum_pos
            
            chrom_end = cum_pos
            chrom_boundaries.append(chrom_end)
            chrom_centers.append((chrom_start + chrom_end) / 2)
            chrom_labels.append(chrom.replace('chr', ''))
            cum_pos += max(plot_data['length']) * 0.02
    
    final_pos = plot_data['cum_end'].max()
    
    # Determine number of subplots based on whether haplotype data is provided
    if hap_count_df is not None and cells is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    
    # Plot cost function with both minima marked
    m_plot_values = list(cost_results.keys())
    costs = list(cost_results.values())
    
    ax1.plot(m_plot_values, costs, 'black', linewidth=2)
    ax1.axvline(x=optimal_m, color='red', linestyle='--', linewidth=2, 
               label=f'm = {optimal_m:.2f}')
    ax1.axvline(x=local_m, color='green', linestyle='--', linewidth=2,
               label=f'm = {local_m:.2f}')
    ax1.set_xlabel('Multiplier (m)')
    ax1.set_ylabel('Weighted Cost')
    ax1.set_title(f'Ploidy Comparison for {pop_col.replace("_ratio", "")}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Create multiple y-axes for copy number plot
    ax2_green_left = ax2.twinx()  # Far left green axis
    ax2_green_right = ax2.twinx()  # Right green axis
    ax2_red_right = ax2.twinx()  # Far right red axis
    
    # Position the axes
    ax2_green_left.yaxis.set_ticks_position('left')
    ax2_green_left.yaxis.set_label_position('left')
    ax2_green_left.spines['left'].set_position(('outward', 60))  # Move far left
    
    ax2_red_right.spines['right'].set_position(('outward', 60))  # Move far right
    
    # Plot the scaled copy numbers
    scaled_col = f'{pop_col}_scaled'
    
    for idx, row in plot_data.iterrows():
        x_coords = [row['cum_start'], row['cum_end']]
        y_coords = [row[scaled_col], row[scaled_col]]
        
        label = 'Scaled Copy Number' if idx == 0 else None
        ax2.plot(x_coords, y_coords, color='black', linewidth=2, alpha=0.8, label=label)
        
        # Connect segments within same chromosome
        if idx > 0:
            prev_row = plot_data.iloc[idx-1]
            if row['chrom'] == prev_row['chrom']:
                ax2.plot([prev_row['cum_end'], row['cum_start']], 
                       [prev_row[scaled_col], row[scaled_col]], color='black', linewidth=1, alpha=0.8)
    
    # Add chromosome boundary lines
    for boundary in chrom_boundaries[:-1]:
        ax2.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    # Calculate scaling factor for comparison
    scale_factor = optimal_m / local_m
    
    # Set y-axis limits first
    all_values = list(plot_data[scaled_col])
    y_max = max(all_values) * 1.1
    #y_min = max(0, min(all_values) * 0.9)
    y_max = min(12, y_max)
    y_min = 0
    
    ax2.set_ylim(y_min, y_max)
    ax2_green_left.set_ylim(y_min, y_max)
    ax2_green_right.set_ylim(y_min, y_max)
    ax2_red_right.set_ylim(y_min, y_max)
    
    # RED reference lines and LEFT axis labels (for optimal_m)
    red_positions = []
    red_labels = []
    for i in range(1, 12):
        if y_min <= i <= y_max:
            ax2.axhline(y=i, color='red', alpha=0.5)
            red_positions.append(i)
            red_labels.append(str(i))
    
    ax2.set_yticks(red_positions)
    ax2.set_yticklabels(red_labels)
    ax2.tick_params(axis='y', colors='red', labelsize=10)
    ax2.set_ylabel(f'Integer CN values with ploidy {optimal_m:.1f}', fontsize=12, color='red')
    
    # GREEN reference lines and positions (for local_m)
    green_positions = []
    green_labels = []
    for i in range(1, 15):
        scaled_pos = i * scale_factor
        if y_min <= scaled_pos <= y_max:
            ax2.axhline(y=scaled_pos, color='green', alpha=0.5)
            green_positions.append(scaled_pos)
            green_labels.append(str(i))
    
    # FAR LEFT green axis
    ax2_green_left.set_yticks(green_positions)
    ax2_green_left.set_yticklabels(green_labels)
    ax2_green_left.tick_params(axis='y', colors='green', labelsize=10)
    ax2_green_left.set_ylabel(f'Integer CN values with ploidy {local_m:.1f}', fontsize=12, color='green')
    
    # RIGHT green axis 
    ax2_green_right.set_yticks(green_positions)
    ax2_green_right.set_yticklabels(green_labels)
    ax2_green_right.tick_params(axis='y', colors='green', labelsize=10)
    ax2_green_right.set_ylabel(f'Integer CN values with ploidy {local_m:.1f}', fontsize=12, color='green')
    
    # FAR RIGHT red axis labels (duplicate of optimal_m for convenience)
    ax2_red_right.set_yticks(red_positions)
    ax2_red_right.set_yticklabels(red_labels)
    ax2_red_right.tick_params(axis='y', colors='red', labelsize=10)
    ax2_red_right.set_ylabel(f'Integer CN values with ploidy {optimal_m:.1f}', fontsize=12, color='red')
    
    # Set main plot properties
    ax2.set_xlabel('Chromosome', fontsize=12)
    ax2.set_title(f'Copy Number Profile Comparison\nmultiplier {optimal_m:.1f} vs. {local_m:.1f}')
    ax2.set_xticks(chrom_centers)
    ax2.set_xticklabels(chrom_labels, rotation=0)
    ax2.set_xlim(0, final_pos)
    ax2.legend(loc='upper left')

    # Add allele frequency plot if haplotype data is provided
    if hap_count_df is not None and cells is not None:
        # Filter hap_count_df for these cells
        pop_hap_df = hap_count_df[hap_count_df['cell'].isin(cells)].copy()
        
        # Compute haplotype ratio for each bin and abspos
        pop_hap_ratio = pop_hap_df.groupby(['bin', 'abspos']).agg({'hap_count':'sum', 'Total':'sum'}).reset_index()
        pop_hap_ratio['hap_ratio'] = pop_hap_ratio['hap_count'] / pop_hap_ratio['Total']
        pop_hap_ratio['folded'] = pop_hap_ratio['hap_ratio'].apply(lambda x: min(x, 1-x))
        
        # Add chromosome information
        bin_to_chrom_dict = dict(zip(hap_count_df[["chrom", "bin"]].drop_duplicates()["bin"].values, 
                                    hap_count_df[["chrom", "bin"]].drop_duplicates()["chrom"].values))
        pop_hap_ratio["chrom"] = pop_hap_ratio["bin"].apply(lambda x: bin_to_chrom_dict[x])
        
        # Sort by chromosome and position
        pop_hap_ratio_sorted = pop_hap_ratio.sort_values(by=['chrom', 'abspos'])
        
        # Plot the folded allele frequency
        ax3.scatter(pop_hap_ratio_sorted['abspos'], pop_hap_ratio_sorted['folded'], s=1, alpha=0.8)
        ax3.set_xlabel('Chromosome', fontsize=12)
        ax3.set_ylabel('Allele Frequency')
        ax3.set_title(f'Allele Frequency for {pop_col.replace("_ratio", "")} - averaged over {len(cells)} cells')
        
        # Add vertical lines for chromosome boundaries in allele frequency plot
        chrom_boundaries_hap = pop_hap_ratio_sorted.groupby('chrom')['abspos'].max()
        for boundary in chrom_boundaries_hap:
            ax3.axvline(x=boundary, color='grey', linestyle='--', alpha=0.5)
        
        # Add chromosome labels at midpoints for allele frequency plot
        chrom_midpoints = pop_hap_ratio_sorted.groupby('chrom')['abspos'].median()
        ax3.set_xticks(chrom_midpoints)
        ax3.set_xticklabels([c[3:] for c in chrom_midpoints.index], rotation=0)
        
        # Set y-axis limits to focus on 0 to 0.5
        ax3.set_ylim(0, 0.5)
        ax3.set_xlim(pop_hap_ratio_sorted['abspos'].min(), pop_hap_ratio_sorted['abspos'].max())
    
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
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    
    # Calculate segment lengths
    segment_lengths = segment_info['end'] - segment_info['start']
    
    # Normalize segment lengths to determine how many rows each segment gets
    # Scale so that smallest segment gets 1 row, largest gets proportional amount
    min_length = segment_lengths.min()
    #max_rows_per_segment = 10  # Maximum rows any single segment can occupy
    
    # Calculate rows per segment based on length (minimum 1 row)
    rows_per_segment = np.maximum(1, np.round(segment_lengths / min_length).astype(int))
    # Cap at maximum to prevent huge chromosomes from dominating
    #rows_per_segment = np.minimum(rows_per_segment, max_rows_per_segment)
    
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
    
    # Create custom colormap: red (deletions) -> white (CN=2) -> blue (amplifications)
    colors_list = [
        (0.0, 'darkred'),     # CN = 0
        (0.25, 'red'),        # CN = 1
        (0.5, 'white'),        # CN = 2 (normal)
        (0.75, 'blue'),         # CN = 3-4
        (1.0, 'darkblue')       # CN = 5+
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

def plot_overlap_matrix_figure(pop1, pop2, name1="A", name2="B", title=None, figsize=(10, 8)):
    """
    Create a colorful overlap matrix figure suitable for presentations
    
    Parameters:
    - pop1, pop2: population dictionaries
    - name1, name2: names for the population sets
    - title: custom title (if None, uses default)
    - figsize: figure size tuple
    
    Returns:
    - reordered_overlap_df: DataFrame with overlap counts (reordered)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    
    # Step 1: Get the basic overlap matrix using existing function
    overlap_df = print_overlap_matrix(pop1, pop2)
    
    # Step 2: Reorder rows based on maximum overlap per column (your logic)
    new_order = []
    for col in overlap_df.columns:
        # Find the index of the maximum value in the column
        max_index = overlap_df[col].idxmax()
        if max_index not in new_order:
            new_order.append(max_index)
    
    # Add any remaining rows that weren't selected
    for row in overlap_df.index:
        if row not in new_order:
            new_order.append(row)
    
    # Reorder the DataFrame
    reordered_overlap_df = overlap_df.reindex(new_order)
    
    # Step 3: Create the visualization with reordered matrix
    overlap_matrix = reordered_overlap_df.values
    pop1_names = reordered_overlap_df.index.tolist()
    pop2_names = reordered_overlap_df.columns.tolist()
    
    # Create custom colormap for better visualization
    colors = ['white', 'lightblue', 'steelblue', 'navy', 'darkblue']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom_blues', colors, N=n_bins)
    
    # Create normalized matrix for color mapping
    color_matrix = np.copy(overlap_matrix)
    color_matrix = np.where(color_matrix > 50, 50, color_matrix)  # Cap at 50
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the main heatmap with custom color scaling
    im = ax.imshow(color_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=50)
    
    # Add text annotations with cell counts
    for i in range(len(pop1_names)):
        for j in range(len(pop2_names)):
            text = int(overlap_matrix[i, j])
            
            # Determine text color based on background
            if overlap_matrix[i, j] < 15:
                text_color = 'black'
            elif overlap_matrix[i, j] < 30:
                text_color = 'white'
            else:
                text_color = 'white'
                
            ax.text(j, i, text, ha='center', va='center', 
                    fontsize=12, fontweight='bold', color=text_color)
    
    # Customize main heatmap
    ax.set_xticks(range(len(pop2_names)))
    ax.set_yticks(range(len(pop1_names)))
    ax.set_xticklabels(pop2_names, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(pop1_names, fontsize=12)
    ax.set_xlabel(f'{name2} Populations', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{name1} Populations', fontsize=14, fontweight='bold')
    
    # Add gridlines
    ax.set_xticks(np.arange(len(pop2_names)+1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(pop1_names)+1)-0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add colorbar with custom labels
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Number of Shared Cells', fontsize=12, fontweight='bold')
    
    # Set custom colorbar ticks
    cbar_ticks = [0, 15, 25, 35, 50]
    cbar_labels = ['0', '15', '25', '35', '50+']
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_labels)
    
    # Main title
    if title is None:
        title = f'Cell Overlap Matrix: {name1} vs {name2} Populations (Reordered)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    return reordered_overlap_df

def plot_uniformity_histograms_nice(all_data, populations, data_lengths, uniformity_data=None, title_prefix="Population", figsize=(12, 8)):
    """
    Create nice histogram plots for uniformity analysis across populations.
    
    Parameters:
    - all_data: DataFrame with segments as rows and cells as columns
    - populations: Dictionary of populations with their cell lists
    - data_lengths: Series with segment lengths
    - uniformity_data: Optional tuple (unif_dict, normal_pop_name) from make_unif_dict().
                      If None, will compute it using make_unif_dict().
    - title_prefix: Prefix for the plot title
    - figsize: Figure size tuple
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from copy_number import make_unif_dict
    
    # Set style for nicer plots
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Get uniformity data if not provided
    if uniformity_data is None:
        print("Computing uniformity data using make_unif_dict...")
        unif_data, normal_pop = make_unif_dict(all_data, populations, data_lengths)
    else:
        unif_data, normal_pop = uniformity_data
        print(f"Using provided uniformity data. Normal population: {normal_pop}")
    
    # Calculate dev_from_median for display purposes
    dev_from_median = {}
    for pop in unif_data:
        dev_from_median[pop] = np.median(np.abs(unif_data[pop] - 1))
    
    # Sort populations by deviation (normal first)
    sorted_pops = sorted(unif_data.keys(), key=lambda x: dev_from_median[x])
    
    # Create subplots
    n_pops = len(sorted_pops)
    n_cols = min(3, n_pops)
    n_rows = (n_pops + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_pops == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Colors for different populations
    colors = plt.cm.Set2(np.linspace(0, 1, n_pops))
    
    for idx, pop in enumerate(sorted_pops):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        unif = unif_data[pop]
        dev = dev_from_median[pop]
        n_cells = len(populations[pop]["cells"])
        
        # Create histogram with better styling
        ax.hist(unif, bins=30, alpha=0.7, color=colors[idx], 
                edgecolor='black', linewidth=0.5, density=True)
        
        # Add vertical line at median - no legend
        median_val = np.median(unif)
        ax.axvline(median_val, color='red', linestyle='--', linewidth=2)
        
        # Styling - simplified title without repetitive normalization info
        ax.set_xlabel('Normalized Coverage', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{pop}\n{n_cells} cells, MAD = {dev:.3f}', 
                     fontsize=12, fontweight='bold')
        ax.set_xlim(0, 5)
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f'Mean: {np.mean(unif):.2f}\nStd: {np.std(unif):.2f}'
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(n_pops, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            if col < len(axes):
                axes[col].set_visible(False)
    
    # Main title with normalization explanation
    plt.suptitle(f'{title_prefix} Uniformity Analysis\n(Coverage normalized by segment length, then by median)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print(f"\n{title_prefix} Uniformity Summary:")
    print("=" * 60)
    print(f"{'Population':<12} {'Cells':<8} {'MAD':<8} {'Mean':<8} {'Std':<8}")
    print("-" * 60)
    
    for pop in sorted_pops:
        unif = unif_data[pop]
        n_cells = len(populations[pop]["cells"])
        dev = dev_from_median[pop]
        mean_val = np.mean(unif)
        std_val = np.std(unif)
        print(f"{pop:<12} {n_cells:<8} {dev:<8.3f} {mean_val:<8.2f} {std_val:<8.2f}")
    
    # Identify normal population
    print(f"\nIdentified normal population: {normal_pop} (lowest MAD = {dev_from_median[normal_pop]:.3f})")
    
    return unif_data, dev_from_median, normal_pop