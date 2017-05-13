#!/bin/bash

img_b="img_background.ppm"
img_t="img_target.ppm"
img_m="img_mask.pgm"
offset_y="130"
offset_x="600"
output="output.ppm"

cd ./lab3_test
time ../main ${img_b} ${img_t} ${img_m} ${offset_y} ${offset_x} ${output}
cd ..
python trans.py
