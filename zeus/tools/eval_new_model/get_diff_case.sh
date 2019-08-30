#!/bin/bash
sampel_file=$1
model=$2
img_dir=$3
labels=$4
diff_dir=$5
sample_num=$6
python download_spark.py ${sampel_file} ${img_dir}
python ../../train/evaluation.py --model ${model} --input ${img_dir} --labels ${labels}
cp inference_output.txt ${img_dir}
cd ${img_dir}
python ../gen_diff.py inference_output.txt ${diff_dir} ${sample_num}
