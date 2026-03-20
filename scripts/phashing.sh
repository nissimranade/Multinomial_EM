
for chr in {1..22} X; do
  echo "Splitting chr$chr..."
  bcftools view -r $chr $INPUT -Oz -o $OUTDIR/t46.var.chr${chr}.vcf.gz
  tabix -p vcf $OUTDIR/t46.var.chr${chr}.vcf.gz
done

for chr in {1..22} X; do
  (
    java -Xmx20g -jar $BEAGLE \
      gt=$OUTDIR/t46.var.chr${chr}.vcf.gz \
      ref=$REFDIR/ALL.chr${chr}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz \
      map=$MAPDIR/plink.chr${chr}.GRCh38.map \
      out=$OUTDIR/t46.var.chr${chr}.phased
  ) &
done

for chr in {1..22} X; do
  echo "Adding chr prefix back for chr$chr..."

  zcat $OUTDIR/t46.var.chr${chr}.phased.vcf.gz | \
  awk 'BEGIN{FS=OFS="\t"}
  /^##contig=/ {gsub("ID=","ID=chr"); print; next}
  /^#/ {print; next}
  {$1="chr"$1; print}' | \
  bgzip > $OUTDIR/t46.var.chr${chr}.phased.withchr.vcf.gz

  tabix -p vcf $OUTDIR/t46.var.chr${chr}.phased.withchr.vcf.gz
done

for chr in {1..22} X; do
  echo "Liftover chr$chr..."

  $GATK LiftoverVcf \
    --INPUT $OUTDIR/t46.var.chr${chr}.phased.withchr.vcf.gz \
    --OUTPUT $OUTDIR/t46.var.chr${chr}.phased.hg19.vcf.gz \
    --CHAIN $CHAIN \
    --REJECT $OUTDIR/t46.var.chr${chr}.rejected.hg19.vcf.gz \
    --REFERENCE_SEQUENCE $HG19FA
done

chr="1"
$GATK LiftoverVcf \
    --INPUT $OUTDIR/t46.var.chr${chr}.phased.withchr.vcf.gz \
    --OUTPUT $OUTDIR/t46.var.chr${chr}.phased.hg19.vcf.gz \
    --CHAIN $CHAIN \
    --REJECT $OUTDIR/t46.var.chr${chr}.rejected.hg19.vcf.gz \
    --REFERENCE_SEQUENCE $HG19FA

