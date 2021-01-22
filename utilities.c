#include <stdio.h>
#include <stdlib.h>
#include <math.h> // sqrt, M_PI
#include <stdbool.h>

double gaussian(double sigma, double x) //best gaussian for patches is sigma=x, where x is 1-based distance from center pixel
{
    return (1 / (sigma * sqrt(2 * M_PI))) * exp(-x * x / (2 * sigma * sigma));
}
double *gaussianWeight(int patchSize)
{ //1-based distance from centre

    double *weights = (double *)malloc(patchSize * sizeof(double));
    double sigma=1/patchSize;
    for (int i = 1; i <= patchSize; i++)
    {
        
        weights[i - 1] = gaussian(i, sigma);
    }
    return weights;
}

bool isPowerOfTwo(int n)
{
    if (n == 0)
        return false;

    return (ceil(log2(n)) == floor(log2(n)));
}

int *readCSV(int *n, char *file)
{
    FILE *matFile;
    matFile = fopen(file, "r");
    if (matFile == NULL)
    {
        printf("Could not open file %s", file);
        exit(-1);
    }
    int size, error;
    size = 1;
    error = 1;
    int *array = (int *)malloc(size * sizeof(int));
    while (error)
    {
        error = fscanf(matFile, "%d,", &array[size - 1]);
        if (error != 1)
        {
            printf("Error reading\n");
            exit(-2);
        }
        size++;
        if (isPowerOfTwo(size))
        {
            array = realloc(array, size * 2 * sizeof(int));
        }
    }
    *n = sqrt(size);

    fclose(matFile);
    return array;
}