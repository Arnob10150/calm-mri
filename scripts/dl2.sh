set -e
B=https://huggingface.co/datasets
cd "/f/Yash_AJSE/work/data/raw/sartaj"
for f in Training.zip Testing.zip; do [ -s "$f" ] || curl -sL --retry 5 -C - -o "$f" "$B/sartajbhuvaji/Brain-Tumor-Classification/resolve/main/$f"; echo "sartaj $f $(stat -c%s $f)"; done
cd "/f/Yash_AJSE/work/data/raw/hemg"
[ -s train.parquet ] || curl -sL --retry 5 -C - -o train.parquet "$B/Hemg/Brain-Tumor-MRI-Dataset/resolve/main/data/train-00000-of-00001.parquet"
echo "hemg $(stat -c%s train.parquet)"
echo "DL2 DONE"
