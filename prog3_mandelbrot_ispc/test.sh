#!/usr/bin/env bash

echo "" > output.txt

for n in {1..40}; do
	sed -i "1s/=.*/=$n;/" mandelbrot.ispc 
	echo "$n tasks ========================================" >> output.txt
	make 
	./mandelbrot_ispc --tasks | tail -n 1 >> output.txt
done

