library(Seurat)
library(patchwork)
library(ggplot2)

# Path to your file
file_path <- "../data/t57/t57.bin_count_table_all_cells.txt"
good_cells_path <- "../data/t57/t57.not_broken_final_list.txt"

# Load full table
df <- read.table(file_path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)

# Load the not broken cells' names
good_cells <- scan(good_cells_path, what = "", sep = "\n")

# Extract bin metadata
bin_info <- df[, 1:5]

# Extract count matrix
counts_all <- df[, -(1:5)]
# Convert to matrix
counts_all <- as.matrix(counts_all)

good_cells <- intersect(good_cells, colnames(counts_all))

counts_good <- counts_all[, good_cells]

#First analyze all cells including broken ones
# Create Seurat object
seurat_obj_all <- CreateSeuratObject(counts = counts_all, project = "T57_Cells", min.cells = 25)

# Normalize and find features
seurat_obj_all <- NormalizeData(seurat_obj_all)
seurat_obj_all <- FindVariableFeatures(seurat_obj_all)

# Scale, PCA, and clustering
seurat_obj_all <- ScaleData(seurat_obj_all)
seurat_obj_all <- RunPCA(seurat_obj_all)
seurat_obj_all <- FindNeighbors(seurat_obj_all, dims = 1:10)
seurat_obj_all <- FindClusters(seurat_obj_all, resolution = 0.5)
seurat_obj_all <- RunUMAP(seurat_obj_all, dims = 1:10)

# WGS bins - all cells cluster assignments
clusters_wgs_all <- data.frame(
  cell_id = colnames(seurat_obj_all),
  cluster = Idents(seurat_obj_all)
)
write.table(clusters_wgs_all, file = "cluster_assignments_WGS_all_cells.txt", 
            sep = "\t", row.names = FALSE, quote = FALSE)

# Save results
saveRDS(seurat_obj_all, file = "seurat_clusters_WGS_all_cells.rds")

# Plot and save
p <- DimPlot(seurat_obj_all, reduction = "umap", label = TRUE)
ggsave("seurat_clusters_WGS_all_cells_umap.png", p, width = 6, height = 5)

# Now analyze only good cells
# Create Seurat object
seurat_obj_good <- CreateSeuratObject(counts = counts_good, project = "T57_Cells", min.cells = 25)

# Normalize and find features
seurat_obj_good <- NormalizeData(seurat_obj_good)
seurat_obj_good <- FindVariableFeatures(seurat_obj_good)

# Scale, PCA, and clustering
seurat_obj_good <- ScaleData(seurat_obj_good)
seurat_obj_good <- RunPCA(seurat_obj_good)
seurat_obj_good <- FindNeighbors(seurat_obj_good, dims = 1:10)
seurat_obj_good <- FindClusters(seurat_obj_good, resolution = 0.5)
seurat_obj_good <- RunUMAP(seurat_obj_good, dims = 1:10)

# WGS bins - good cells cluster assignments
clusters_wgs_good <- data.frame(
  cell_id = colnames(seurat_obj_good),
  cluster = Idents(seurat_obj_good)
)
write.table(clusters_wgs_good, file = "cluster_assignments_WGS_good_cells.txt",
            sep = "\t", row.names = FALSE, quote = FALSE)


# Save results
saveRDS(seurat_obj_good, file = "seurat_clusters_WGS_good_cells.rds")

# Plot and save
p <- DimPlot(seurat_obj_good, reduction = "umap", label = TRUE)
ggsave("seurat_clusters_WGS_good_cells_umap.png", p, width = 6, height = 5)

# Load EM population assignments
em_assignments <- read.table("../data/t57/t57_population_assignments_A_WGS_bins.txt", 
                             header = TRUE, sep = "\t", stringsAsFactors = FALSE)

# Match EM assignments to Seurat cells
# Create a named vector for easy lookup
em_pops <- setNames(em_assignments$population, em_assignments$cell)

# Add EM population to Seurat metadata
seurat_obj_good$EM_population <- em_pops[colnames(seurat_obj_good)]

# Plot UMAP colored by EM populations
p_em <- DimPlot(seurat_obj_good, reduction = "umap", group.by = "EM_population", label = TRUE)
ggsave("seurat_umap_WGS_good_cells_colored_by_EM.png", p_em, width = 6, height = 5)

# Optional: Create side-by-side comparison
p_seurat <- DimPlot(seurat_obj_good, reduction = "umap", label = TRUE) + ggtitle("Seurat Clusters")
p_em_comp <- DimPlot(seurat_obj_good, reduction = "umap", group.by = "EM_population", label = TRUE) + ggtitle("EM Populations")
p_combined <- p_seurat | p_em_comp
ggsave("seurat_vs_EM_comparison_WGS_good_cells.png", p_combined, width = 12, height = 5)


# Path to your file
file_path <- "../data/t57/t57.bincounts_without_WGS_segmentation.txt"


# Load full table
df <- read.table(file_path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)

# Load the not broken cells' names
good_cells <- scan(good_cells_path, what = "", sep = "\n")

# Extract bin metadata
bin_info <- df[, 1:5]

# Extract count matrix
counts_all <- df[, -(1:5)]
# Convert to matrix
counts_all <- as.matrix(counts_all)

good_cells <- intersect(good_cells, colnames(counts_all))

counts_good <- counts_all[, good_cells]

#First analyze all cells including broken ones
# Create Seurat object
seurat_obj_all <- CreateSeuratObject(counts = counts_all, project = "T57_Cells", min.cells = 25)

# Normalize and find features
seurat_obj_all <- NormalizeData(seurat_obj_all)
seurat_obj_all <- FindVariableFeatures(seurat_obj_all)

# Scale, PCA, and clustering
seurat_obj_all <- ScaleData(seurat_obj_all)
seurat_obj_all <- RunPCA(seurat_obj_all)
seurat_obj_all <- FindNeighbors(seurat_obj_all, dims = 1:10)
seurat_obj_all <- FindClusters(seurat_obj_all, resolution = 0.5)
seurat_obj_all <- RunUMAP(seurat_obj_all, dims = 1:10)

# equi bins - all cells cluster assignments
clusters_equi_all <- data.frame(
  cell_id = colnames(seurat_obj_all),
  cluster = Idents(seurat_obj_all)
)
write.table(clusters_equi_all, file = "cluster_assignments_equi_bins_all_cells.txt", 
            sep = "\t", row.names = FALSE, quote = FALSE)


# Save results
saveRDS(seurat_obj_all, file = "seurat_clusters_equi_bins_all_cells.rds")

# Plot and save
p <- DimPlot(seurat_obj_all, reduction = "umap", label = TRUE)
ggsave("seurat_clusters_equi_bins_all_cells_umap.png", p, width = 6, height = 5)

# Now analyze only good cells
# Create Seurat object
seurat_obj_good <- CreateSeuratObject(counts = counts_good, project = "T57_Cells", min.cells = 25)

# Normalize and find features
seurat_obj_good <- NormalizeData(seurat_obj_good)
seurat_obj_good <- FindVariableFeatures(seurat_obj_good)

# Scale, PCA, and clustering
seurat_obj_good <- ScaleData(seurat_obj_good)
seurat_obj_good <- RunPCA(seurat_obj_good)
seurat_obj_good <- FindNeighbors(seurat_obj_good, dims = 1:10)
seurat_obj_good <- FindClusters(seurat_obj_good, resolution = 0.5)
seurat_obj_good <- RunUMAP(seurat_obj_good, dims = 1:10)

# equi bins - good cells cluster assignments
clusters_equi_good <- data.frame(
  cell_id = colnames(seurat_obj_good),
  cluster = Idents(seurat_obj_good)
)
write.table(clusters_equi_good, file = "cluster_assignments_equi_bins_good_cells.txt", 
            sep = "\t", row.names = FALSE, quote = FALSE)


# Save results
saveRDS(seurat_obj_good, file = "seurat_clusters_equi_bins_good_cells.rds")

# Plot and save
p <- DimPlot(seurat_obj_good, reduction = "umap", label = TRUE)
ggsave("seurat_clusters_equi_bins_good_cells_umap.png", p, width = 6, height = 5)

