# GPGPU Lab3
## Possion Editing with Jacobi Iteration
### run.sh
```
./run.sh [0 or 1]
```
* 0 is for sample testcase
* 1 is for my own testcase

### trans.py
* convert output.ppm to output.png
* note that *run.sh* would call it.

### lab3.cu
* *lab3_baseline.cu* use the original way to do the iteration.
    * converges at 20000 iterations.
* *lab3_laplacian.cu* use the normal laplacian 5x5 coef. matrix
    * converges at 10000 iterations.
* *lab3.cu* use the advanced laplacian 5x5 coef. matrix
    * converges at 8000 iterations.
