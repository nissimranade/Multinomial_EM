import pysam
import pandas as pd
import sys


bamfilelist_file = sys.argv[1]
variant_file = sys.argv[2]
output_file = sys.argv[3]
tumor = sys.argv[4]
#bamfilelist_file = /mnt/wigstore6/user/ranade/binless_clustering/DNA_alone/{tumor}/{tumor}.bamfilelist_from_siran.txt
#output_file = f"../data/{tumor}/{tumor}.tumor_spec_variant_counts.txt"
#variant_file = f"/mnt/wigstore6/user/ranade/binless_clustering/DNA_alone/{tumor}/tumor_specific_variants.txt"
variant_list = pd.read_csv(variant_file, sep="\t", header = None)

bincount_table = pd.read_csv(f"../data/{tumor}/{tumor}.bin_count_table_all_cells.txt", sep = "\t")

cell_list = bincount_table.columns[5:]

bamfilelist = pd.read_csv(bamfilelist_file, header = None)[0].values
results = []

for bam_path in bamfilelist:
    bam = pysam.AlignmentFile(bam_path, "rb")
    min_base_quality = 20
    for variant in variant_list.values:
        print(variant)
        chrom, pos_start, pos_end, ref, alt = variant
        uniq_ids = set()
        cell_id_specific_counts = {}
        for pileup_column in bam.pileup(
                        chrom,
                        pos_start,
                        pos_end,
                        truncate=True,
                        stepper="all"):
            if pileup_column.pos != pos_start:
                continue
            count = 0
            for read in pileup_column.pileups:

                if read.is_del or read.is_refskip:
                    continue

                base = read.alignment.query_sequence[read.query_position]
                qual = read.alignment.query_qualities[read.query_position]

                if qual < min_base_quality:
                    continue

                count += 1
                read_id = read.alignment.query_name
                tag_info = read_id.split("_")
                cell_id = tag_info[1]
                molecule_id = tag_info[2]
                id = cell_id+"_"+molecule_id
                if id in uniq_ids:
                    continue
                uniq_ids.add(id)
                if cell_id not in cell_id_specific_counts:
                    cell_id_specific_counts[cell_id] = {
                        "A_count": 0,
                        "T_count": 0,
                        "C_count": 0,
                        "G_count": 0,
                        "ref_count": 0,
                        "alt_count": 0,
                        "total_count": 0
                    }

                if base == "A":
                    cell_id_specific_counts[cell_id]["A_count"] += 1
                elif base == "T":
                    cell_id_specific_counts[cell_id]["T_count"] += 1
                elif base == "C":
                    cell_id_specific_counts[cell_id]["C_count"] += 1
                elif base == "G":
                    cell_id_specific_counts[cell_id]["G_count"] += 1
                if base == ref:
                    cell_id_specific_counts[cell_id]["ref_count"] += 1
                if base == alt:
                    cell_id_specific_counts[cell_id]["alt_count"] += 1
                cell_id_specific_counts[cell_id]["total_count"] += 1
        

        for cell in cell_id_specific_counts:
            if cell_id_specific_counts[cell]["total_count"] > 0:
                if cell in cell_list:
                    results.append({"chrom": chrom, "pos": pos_end, "cell_id": cell, **cell_id_specific_counts[cell]})

pd.DataFrame(results).to_csv(output_file, sep="\t", index=False)