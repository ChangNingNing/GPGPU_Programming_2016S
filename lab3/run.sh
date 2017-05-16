#!/bin/bash

img_b=("img_background.ppm" "beach.ppm")
img_t=("img_target.ppm" "giant.ppm")
img_m=("img_mask.pgm" "giant_mask.pgm")
offset_y=("130" "420")
offset_x=("600" "90")
output="output.ppm"

cd ./lab3_test
time ../main ${img_b[$1]} ${img_t[$1]} ${img_m[$1]} ${offset_y[$1]} ${offset_x[$1]} ${output}
#perf stat ../main ${img_b[$1]} ${img_t[$1]} ${img_m[$1]} ${offset_y[$1]} ${offset_x[$1]} ${output}
cd ..
python trans.py
