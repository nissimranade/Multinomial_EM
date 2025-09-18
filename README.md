# Single-Cell Copy Number Analysis with Recursive EM Clustering

A comprehensive pipeline for analyzing copy number variations in single-cell sequencing data using recursive Expectation-Maximization (EM) clustering with slope-based validation.

## Overview

This repository implements a novel method for discovering cell populations and inferring copy number profiles from single-cell DNA sequencing data. The key innovation is using **ratio-of-ratios analysis with RANSAC regression** to validate population splits and prevent over-clustering.

## Key Features

### 🧬 **Recursive EM Population Discovery**
- Automatically discovers cell populations using multinomial EM algorithm
- **Slope-based validation**: Rejects spurious splits using ratio-of-ratios analysis
- **RANSAC regression**: Robust slope estimation to handle outliers
- **Chi-squared selection**: Identifies most informative genomic segments

### 📊 **Copy Number Analysis**
- **Optimal ploidy estimation**: Finds best integer copy number profiles using weighted least squares
- **Population reassignment**: Refines clusters based on maximum likelihood
- **Normal population identification**: Automatically detects diploid cells

### 📈 **Comprehensive Visualization**
- **Step function plots**: Copy number profiles across chromosomes
- **Population heatmaps**: Genomic-proportional visualization
- **Ploidy optimization plots**: Cost function visualization
- **Population comparison**: Overlap matrices and pairwise analysis

## Quick Start

1. **Place your data files in the `data/` directory:**
   ```
   data/
   ├── t57.bin_count_table_all_cells.txt
   ├── t57.not_broken_final_list.txt
   ├── enca57_1M.hg19.5k.k50.varbin.short.cbs.txt
   └── normal_cells.txt
   ```

2. **Run the analysis:**
   Open `notebooks/new_beginning_v2.ipynb` and run all cells sequentially.

## Algorithm Pipeline

### 1. **Data Loading and Preprocessing**
- Load bin count tables, cell lists, and breakpoint data
- Filter out chrY and create segment length information

### 2. **Recursive EM Clustering**
```python
final_populations = recursive_em_split(
    all_data, cells, threshold=3/4, max_depth=10, current_depth=0, name="A"
)
```
- Starts with all cells as one population
- Uses multinomial EM to split into two clusters
- **Validates splits** using ratio-of-ratios analysis:
  - Selects top 10 most different segments (by chi-squared)
  - Computes ratios against bottom 30 most similar segments
  - Fits RANSAC regression lines through ratio scatter plots
  - **Rejects split** if all slopes fall between threshold and 1/threshold

### 3. **Population Refinement**
```python
reassigned_populations = reassign_cells(all_data, final_populations)
```
- Reassigns cells based on maximum likelihood under learned parameters
- Identifies normal population by uniformity across segments

### 4. **Copy Number Analysis**
```python
# Compute population-average ratios to normal
avg_counts, ratios_to_normal, normal_pop = compute_average_segment_counts(
    all_data, reassigned_populations
)

# Optimize ploidy for each population
optimal_m, min_cost, profile_df, costs = compute_optimal_ploidy_and_profile(
    ratios_to_normal, pop_col, m_range=[1.5, 6], step_size=0.1
)
```

### 5. **Visualization**
```python
# Plot copy number profiles
plot_copy_number_step_function(ratios_to_normal, pop_cols, title)

# Create population-ordered heatmap
plot_copy_number_heatmap(heatmap_df, boundaries, segment_info, title)
```

## Input Data Format

The pipeline expects:
- **Bin count table**: Segments × cells matrix with read counts
- **Cell list**: Names of cells to include in analysis
- **Breakpoint data**: Chromosome, start, end positions for segments
- **Normal cells** (optional): Known diploid cells for validation

## Key Parameters

- `threshold`: Slope threshold for split validation (default: 3/4)
- `max_depth`: Maximum recursion depth (default: 10)
- `m_range`: Ploidy multiplier search range (default: [1.5, 6])
- `min_population_size`: Minimum cells per population (default: 50)

## Results

The analysis produces:
- **Population assignments**: Cell-to-population mapping
- **Copy number profiles**: Integer copy numbers for each population
- **Ploidy estimates**: Optimal multipliers for each population
- **Visualization plots**: Step functions, heatmaps, and comparisons

## Example Output

```
Found 5 populations:
A111: 234 cells
A112: 156 cells  
A121: 89 cells
A122: 198 cells
A2: 445 cells

Normal population identified as: A2
Optimal multiplier for A111: 2.3 (average CN: 2.7)
Optimal multiplier for A112: 3.1 (average CN: 3.8)
```

## Repository Structure

```
Multinomial_EM/
├── notebooks/
│   └── new_beginning_v2.ipynb    # Complete analysis pipeline
├── data/                         # Input data files
│   ├── t57.bin_count_table_all_cells.txt
│   ├── t57.not_broken_final_list.txt
│   ├── enca57_1M.hg19.5k.k50.varbin.short.cbs.txt
│   └── normal_cells.txt
└── README.md                     # This file
```

## Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import logsumexp
from sklearn.linear_model import RANSACRegressor, LinearRegression
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
```

## License

MIT License - see LICENSE file for details.
