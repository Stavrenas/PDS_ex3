#include <stdio.h>
#include <stdlib.h>
#include <math.h> // sqrt, M_PI
#include <stdbool.h>
#include <time.h>
#include <string.h>

double gaussian(double sigma, double x) //best gaussian for patches is sigma=x, where x is 1-based distance from center pixel
{
    return (1 / (sigma * sqrt(2 * M_PI))) * exp(-x * x / (2 * sigma * sigma));
}
double *gaussianWeight(int patchSize)
{ //1-based distance from centre

    double *weights = (double *)malloc(patchSize * sizeof(double));
    double sigma = 1 / patchSize;
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

int *readCSV(int *n, char *file) //n represents total number of pixels
{
    FILE *matFile;
    matFile = fopen(file, "r");
    if (matFile == NULL)
    {
        printf("Could not open file %s", file);
        exit(-1);
    }
    int pixels, error;
    pixels = 1;
    error = 1;
    int *array = (int *)malloc(pixels * sizeof(int));
    while (error)
    {
        error = fscanf(matFile, "%d,", &array[pixels - 1]);
        //printf("error is %d\n", error);
        if (error != 1)
        {
            printf("finished reading\n");
            *n = sqrt(pixels);
            fclose(matFile);
            return array;
        }
        pixels++;
        //printf("size is %d\n", size);
        if (isPowerOfTwo(pixels))
        {
            array = realloc(array, pixels * 2 * sizeof(int));
        }
    }
    *n = sqrt(pixels);

    fclose(matFile);
    return array;
}

double *normalizeImage(int *image, int size)
{ //size represents the dimension
    int max = 0;
    for (int i = 0; i < size * size; i++)
    {
        if (image[i] > max)
            max = image[i];
    }
    double *array = (double *)malloc(size * size * sizeof(double));
    for (int i = 0; i < size * size; i++)
    {
        array[i] = ((double)image[i]) / max;
    }
    printf("exited normal\n");
    return array;
}

double *addNoiseToImage(double *image, int size)
{
    srand(time(NULL));
    double *noisy = (double *)malloc(size * size * sizeof(double));

    double random_value,effect;
    for (int i = 0; i < size * size; i++)
    {
        random_value = ((double)rand() / RAND_MAX * 20 - 10);
        effect =gaussian(2, random_value) - 0.1;
        noisy[i] = (effect +1) * image[i];
        if (noisy[i] < 0)
            noisy[i] = 0;
        else if (noisy[i] > 1)
            noisy[i] = 1;
    }
    printf("exited noisy\n");
    return noisy;
}

void writeToCSV(double *image, int size, char *name)
{
    FILE *filepointer;
    char *filename = (char *)malloc((strlen(name) + 4) * sizeof(char));
    sprintf(filename, "%s.csv",name);
    filepointer = fopen(filename, "w"); //create a file
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            fprintf(filepointer, "%f,", image[i * size + j]);

        fprintf(filepointer, "\n");
    }
}

double findMax(double * array, int size ){
    double max = 0;
    for (int i = 0; i < size; i++)
    {
        if (array[i] > max)
            max = array[i];
    }
    return max;
}