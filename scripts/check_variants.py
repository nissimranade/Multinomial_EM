import pysam
import pandas as pd
import sys


#bamfilelist_file = sys.argv[1]
# variant_file = sys.argv[2]
# output_file = sys.argv[3]
tumor = sys.argv[1]
chr = sys.argv[2]
bamfilelist_file = f"/mnt/wigstore6/user/ranade/binless_clustering/DNA_alone/{tumor}/{tumor}.bamfilelist_from_siran.txt"
output_file = f"/mnt/wigstore6/user/ranade/binless_clustering/DNA_alone/{tumor}/{tumor}.{chr}.germline_variant_counts.txt"
variant_file = f"/mnt/wigstore6/user/ranade/binless_clustering/DNA_alone/{tumor}/germline_var/{tumor}.var.{chr}.withGT.bed"
variant_list = pd.read_csv(variant_file, sep="\t", header = None)

variant_list = variant_list[(variant_list[3].str.len() == 1) & (variant_list[4].str.len() == 1)]

variant_list = variant_list[(variant_list[3].str.len() == 1) & (variant_list[4].str.len() == 1)]
variant_list = variant_list[variant_list[0] == chr]
variant_list = variant_list[variant_list[5] != "0|0"]
variant_list = variant_list[variant_list[5] != "1|1"]

bincount_table = pd.read_csv(f"../data/{tumor}/{tumor}.bin_count_table_all_cells.txt", sep = "\t")

cell_list = bincount_table.columns[5:]

bamfilelist = pd.read_csv(bamfilelist_file, header = None)[0].values
outfile = open(output_file, "w")
header = "chrom\tpos\tcell_id\tA_count\tT_count\tC_count\tG_count\tref_count\talt_count\ttotal_count\tGT\n"
outfile.write(header)

for bam_path in bamfilelist:
    bam = pysam.AlignmentFile(bam_path, "rb")
    min_base_quality = 20
    for variant in variant_list.values:
        print(variant)
        chrom, pos_start, pos_end, ref, alt, gt = variant
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
                        "total_count": 0,
                        "GT" : gt
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
                    counts = cell_id_specific_counts[cell]
                    line = f"{chrom}\t{pos_end}\t{cell}\t{counts['A_count']}\t{counts['T_count']}\t{counts['C_count']}\t{counts['G_count']}\t{counts['ref_count']}\t{counts['alt_count']}\t{counts['total_count']}\t{counts['GT']}\n"
                    outfile.write(line)

outfile.close()