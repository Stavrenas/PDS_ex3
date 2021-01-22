#include <stdio.h>
#include <stdlib.h>
#include <math.h> // sqrt, M_PI
#include <stdbool.h>
#include"utilities.h"



int main(int argc, char **argv[])
{
    int patchSize = 5;
    double *weights = (double *)malloc(patchSize * sizeof(double));
    weights = gaussianWeight(patchSize);
    for (int i = 0; i < patchSize; i++)
        printf("weight no %d is %f\n", i, weights[i]);
}