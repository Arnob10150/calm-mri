cd "/f/Yash_AJSE/work/data/raw/cheng"
for pair in "3381290:p1.zip" "3381296:p2.zip" "3381293:p3.zip" "3381302:p4.zip"; do
  id="${pair%%:*}"; out="${pair##*:}"
  ( curl -sL --retry 8 --retry-delay 3 -C - -o "$out" "https://ndownloader.figshare.com/files/$id" && echo "OK $out" ) &
done
( curl -sL --retry 5 -C - -o cvind.mat "https://ndownloader.figshare.com/files/7005344"; curl -sL --retry 5 -C - -o README.txt "https://ndownloader.figshare.com/files/51340418"; echo "OK meta" ) &
cd "/f/Yash_AJSE/work/data/raw/sartaj"
B=https://huggingface.co/datasets
for f in Training.zip Testing.zip; do ( curl -sL --retry 5 -C - -o "$f" "$B/sartajbhuvaji/Brain-Tumor-Classification/resolve/main/$f" && echo "OK sartaj-$f" ) & done
cd "/f/Yash_AJSE/work/data/raw/hemg"
( curl -sL --retry 5 -C - -o train.parquet "$B/Hemg/Brain-Tumor-MRI-Dataset/resolve/main/data/train-00000-of-00001.parquet" && echo "OK hemg" ) &
wait
echo "ALL PARALLEL DONE"
