#!/bin/bash

img_b=("img_background.ppm" "beach.ppm")
img_t=("img_target.ppm" "giant.ppm")
img_m=("img_mask.pgm" "giant_mask.pgm")
offset_y=("130" "420")
offset_x=("600" "90")
output=("output" "output_giant")

make
for((i=0; i<2; i+=1))
do
	cd ./lab3_test
	echo ${output[$i]}
	time ../main ${img_b[$i]} ${img_t[$i]} ${img_m[$i]} ${offset_y[$i]} ${offset_x[$i]} "${output[$i]}.ppm"
	cd ..
	python trans.py "lab3_test/${output[$i]}.ppm" "lab3_test/${output[$i]}.png"
done
