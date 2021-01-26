#include <stdio.h>
#include <stdlib.h>
#include <math.h> // sqrt, M_PI
#include <stdbool.h>
#include <time.h>
#include <string.h>

double gaussian(double sigma, double x)
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
        if (error != 1)
        {
            printf("finished reading\n");
            *n = sqrt(pixels);
            fclose(matFile);
            return array;
        }
        pixels++;
        array = realloc(array, pixels * sizeof(int));
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
    printf("Finished normalizing\n");
    return array;
}

double *addNoiseToImage(double *image, int size)
{
    srand(time(NULL));
    double *noisy = (double *)malloc(size * size * sizeof(double));

    double random_value, effect;
    for (int i = 0; i < size * size; i++)
    {
        random_value = ((double)rand() / RAND_MAX * 20 - 10);
        effect = gaussian(2, random_value) - 0.05;
        noisy[i] = (effect + 1) * image[i];
        if (noisy[i] < 0)
            noisy[i] = 0;
        else if (noisy[i] > 1)
            noisy[i] = 1;
    }
    printf("Finished adding noise\n");
    return noisy;
}

void writeToCSV(double *image, int size, char *name)
{
    FILE *filepointer;
    char *filename = (char *)malloc((strlen(name) + 4) * sizeof(char));
    sprintf(filename, "%s.csv", name);
    filepointer = fopen(filename, "w"); //create a file
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            fprintf(filepointer, "%f,", image[i * size + j]);

        fprintf(filepointer, "\n");
    }
}

double findMax(double *array, int size)
{
    double max = 0;
    for (int i = 0; i < size; i++)
    {
        if (array[i] > max)
            max = array[i];
    }
    return max;
}

double **createPatches(double *image, int size, int patchSize)
{
    //We assume that patchSize is an odd number//
    //In order to create the patches we must consider that the pixels are stored in Row-Major format//
    //A simple aproach is to handle the patches also in the same format//
    int patchLimit = (patchSize-1) / 2;
    int patchIterator, imageIterator;
    double **patches = (double **)malloc(size * size * sizeof(double *));

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++) //go to each pixel of the image
        {
            int mod = j % size;
            double *patch = (double *)malloc(patchSize * patchSize * sizeof(double)); //We assume that (i,j) is the pixel on the centre
            for (int k = -patchLimit; k <= patchLimit; k++)
            {
                for (int m = -patchLimit; m <= patchLimit; m++) //go to each pixel of the patch: i*size +j
                {
                    patchIterator = (k + patchLimit) * patchSize + (m + patchLimit);
                    imageIterator = (i + k) * size + (j + m);
                    patch[patchIterator] = -1;

                    if (imageIterator >= 0 && imageIterator < size * size) //filter out of image pixels
                    {

                        if (!(mod < patchLimit && m < -mod) && !(mod >= size - patchLimit && m >= size - mod))
                            //!(j % size < patchLimit && m +  < 0) filters pixels that are on the left side of the patch
                            //!(j % size >= size - patchLimit && m  >=size - j % size) filters pixels that are on the right side of the patch
                            patch[patchIterator] = image[imageIterator];
                        //printf("patch: %d , image: %d, m is %d, size - mod is %d\n", patchIterator, imageIterator, m, (size - mod));
                    }
                }
            }
            patches[i * size + j] = (double *)malloc(patchSize * patchSize * sizeof(double));
            patches[i * size + j] = patch;
        }
    }
    return patches;
}

double calculateGaussianDistance(double *patch1, double *patch2, int patchSize, double sigma)
{

    int patchLimit = (patchSize-1) / 2;
    double sum, result, gauss = 0;
    sum = 0;
    double *gaussianWeights = (double *)malloc((patchSize + patchLimit) * sizeof(double));
    for (int i = 0; i < patchSize + patchLimit; i++)
        gaussianWeights[i] = gaussian(sigma, i);

    for (int k = -patchLimit; k <= patchLimit; k++)
    {
        for (int m = -patchLimit; m <= patchLimit; m++) //go to each pixel of the patch: i*size +j
        {
            int patchIterator = (k + patchLimit) * patchSize + (m + patchLimit);
            if (patch1[patchIterator] != -1 && patch2[patchIterator] != -1) //this means out of bounds
            {
                int distance = m * m + k * k; //distance from centre pixel
                double gaussianEffect = gaussianWeights[distance];
                result = (patch1[patchIterator] - patch2[patchIterator]) * (patch1[patchIterator] - patch2[patchIterator]) * gaussianEffect;
                //printf("Between %f and %f : result is %f, distance is %d, gauss is %f, sum is %f\n", patch1[patchIterator], patch2[patchIterator], result,distance, gaussianEffect, sum);
                sum += result;
                //gauss += gaussianEffect;
            }
        }
    }
    //printf("Sum is %f\n", sum);
    //free(gaussianWeights);
    return sum;
}

void printPatch(double *patch, int patchSize)
{

    for (int i = 0; i < patchSize; i++)
    {
        for (int j = 0; j < patchSize; j++)
        {
            if (patch[i * patchSize + j] == -1)
                printf("    x    ");
            else
                printf("%f ", patch[i * patchSize + j]);
        }
        printf("\n");
    }
}

double *denoiseImage(double *image, int size, int patchSize, double sigmaDist, double sigmaGauss)
{
    int totalPixels = size * size;
    double **patches = (double **)malloc(totalPixels * sizeof(double *));
    double **distances = (double **)malloc(totalPixels * sizeof(double *));
    patches = createPatches(image, size, patchSize);

    printf("Finished allocating memory");

    for (int i = 0; i < totalPixels; i++)
    {
        double normalFactor = 0;
        distances[i] = (double *)malloc(totalPixels * sizeof(double));

        for (int j = 0; j < totalPixels; j++)
        {
            double dist = calculateGaussianDistance(patches[i], patches[j], patchSize, sigmaGauss); //calculate distances from each patch with gaussian weights
            //printf("i is %d, j is %d, dist is %f \n\n", i, j, dist);
            distances[i][j] = exp(-dist / (sigmaDist * sigmaDist));
            normalFactor += distances[i][j]; //calculate factor to normalize distances ~ Z[i]
        }
        //printf("NormalFactor: %f\n", normalFactor);

        for (int j = 0; j < totalPixels; j++)
            distances[i][j] /= normalFactor; //distances represents the weight factor for each pixel ~ w(i,j)
    }

    double *denoisedImage = (double *)malloc(totalPixels * sizeof(double));
    for (int i = 0; i < totalPixels; i++)
    {
        denoisedImage[i] = 0;
        for (int j = 0; j < totalPixels; j++)
            denoisedImage[i] += distances[i][j] * image[j];
    }
    printf("Finished denoising \n");
    for (int i = 0; i < totalPixels; i++){
        free(patches[i]);
        free(distances[i]);
    }
    free(patches);
    free(distances);
    return denoisedImage;
}

double *findRemoved(double *noisy, double *denoised, int size)
{
    int totalPixels = size * size;
    double *removed = (double *)malloc(totalPixels * sizeof(double));
    for (int i = 0; i < totalPixels; i++)
        removed[i] = denoised[i] - noisy[i];
    printf("Finished finding removed\n");
    return removed;
}
