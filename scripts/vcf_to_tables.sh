echo -e "CHROM\tPOS\tREF\tALT\tGT_blood\tGT_tumor\tADref_blood\tADref_tumor\tADalt_blood\tADalt_tumor\tAF_blood\tAF_tumor" \
> all_variants.txt

bcftools query \
  -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT\t]\t[%AD{0}\t]\t[%AD{1}\t]\t[%AF{0}]\n' \
  /mnt/wigstore6/user/ranade/binless_clustering/DNA_alone/t57/ENCA_NWH_057.lifted.hg19.vcf.gz \
>> all_variants.txt