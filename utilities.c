#include <stdio.h>
#include <stdlib.h>
#include <math.h> // sqrt, M_PI
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include "utilities.h"

struct timeval tic()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv;
}

double toc(struct timeval begin)
{
    struct timeval end;
    gettimeofday(&end, NULL);
    double stime = ((double)(end.tv_sec - begin.tv_sec) * 1000) +
                   ((double)(end.tv_usec - begin.tv_usec) / 1000);
    stime = stime / 1000;
    return (stime);
}

int *readCSV(int *n, char *file) //n represents total number of pixels
{
    FILE *matFile;
    matFile = fopen(file, "r");
    if (matFile == NULL)
    {
        printf("Could not open file %s\n", file);
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
            //printf("Finished reading image \n");
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

void writeToCSV(float *image, int size, char *name)
{
    FILE *filepointer;
    char *filename = (char *)malloc((strlen(name) + 4) * sizeof(char));
    sprintf(filename, "%s.csv", name);
    filepointer = fopen(filename, "w"); //create a file
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            fprintf(filepointer, "%.6f,", image[i * size + j]); //write each pixel value

        fprintf(filepointer, "\n");
    }
}

float gaussian(float sigma, float x)
{
    return (1 / (sigma * sqrt(2 * M_PI))) * exp(-x * x / (2 * sigma * sigma));
}

float findMax(float *array, int size)
{
    float max = 0;
    for (int i = 0; i < size; i++)
    {
        if (array[i] > max)
            max = array[i];
    }
    return max;
}

float *normalizeImage(int *image, int size) //size represents the dimension
{
    int max = 0;
    for (int i = 0; i < size * size; i++)
    {
        if (image[i] > max)
            max = image[i];
    }
    float *array = (float *)malloc(size * size * sizeof(float));
    for (int i = 0; i < size * size; i++)
    {
        array[i] = ((float)image[i]) / max;
    }
    printf("Finished normalizing\n");
    return array;
}

float *addNoiseToImage(float *image, int size)
{
    srand(time(NULL));
    float *noisy = (float *)malloc(size * size * sizeof(float));

    float random_value, effect;
    for (int i = 0; i < size * size; i++)
    {
        random_value = ((float)rand() / RAND_MAX * 20 - 10);
        effect = gaussian(2, random_value) - 0.05;
        noisy[i] = (effect * 0.5 + 1) * image[i]; //adds gaussian noise
        if (noisy[i] < 0)
            noisy[i] = 0;
        else if (noisy[i] > 1)
            noisy[i] = 1;
    }
    printf("Finished adding noise\n");
    return noisy;
}

float **createPatches(float *image, int size, int patchSize)
{
    //We assume that patchSize is an odd number
    //In order to create the patches we must consider that the pixels are stored in Row-Major format
    //A simple aproach is to handle the patches also in the same format
    int patchLimit = (patchSize - 1) / 2;
    int patchIterator, imageIterator;
    float **patches = (float **)malloc(size * size * sizeof(float *));

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++) //go to each pixel of the image
        {
            float *patch = (float *)malloc(patchSize * patchSize * sizeof(float)); //We assume that (i,j) is the pixel on the centre
            for (int k = -patchLimit; k <= patchLimit; k++)
            {
                for (int m = -patchLimit; m <= patchLimit; m++) //go to each pixel of the patch: i*size +j
                {
                    patchIterator = (k + patchLimit) * patchSize + (m + patchLimit);
                    imageIterator = (i + k) * size + (j + m);
                    patch[patchIterator] = -1;

                    if (imageIterator >= 0 && imageIterator < size * size) //filter out of image pixels
                    {

                        if (!(j < patchLimit && m < -j) && !(j >= size - patchLimit && m >= size - j))
                            //!(j  < patchLimit && m +  < 0) filters pixels that are on the left side of the patch
                            //!(j  >= size - patchLimit && m  >=size - j ) filters pixels that are on the right side of the patch
                            patch[patchIterator] = image[imageIterator];
                    }
                }
            }
            patches[i * size + j] = patch;
        }
    }
    return patches;
}

float calculateGaussianDistance(float *patch1, float *patch2, int patchSize, float *gaussianWeights)
{
    int patchLimit = (patchSize - 1) / 2;
    float sum = 0;

    for (int k = -patchLimit; k <= patchLimit; k++)
    {
        for (int m = -patchLimit; m <= patchLimit; m++) //go to each pixel of the patch: i*size +j
        {
            int patchIterator = (k + patchLimit) * patchSize + (m + patchLimit);
            if (patch1[patchIterator] != -1 && patch2[patchIterator] != -1) //this means out of bounds
            {
                int distance = m * m + k * k; //distance from centre pixel
                sum += (patch1[patchIterator] - patch2[patchIterator]) * (patch1[patchIterator] - patch2[patchIterator]) * gaussianWeights[distance];
            }
        }
    }
    return abs(sum);
}

void printPatch(float *patch, int patchSize)
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

float *denoiseImage(float *image, int size, int patchSize, float sigmaDist, float sigmaGauss)
{
    int totalPixels = size * size;
    int patchLimit = (patchSize - 1) / 2;
    float **patches, *gaussianWeights;

    gaussianWeights = (float *)malloc((patchSize + patchLimit) * sizeof(float));

    patches = createPatches(image, size, patchSize); //patch creation

    for (int i = 0; i < patchSize + patchLimit; i++)
        gaussianWeights[i] = gaussian(sigmaGauss, i); //calculate all the necessary gaussian weights

    float *denoisedImage = (float *)malloc(totalPixels * sizeof(float));
    float dist,normalFactor;

    for (int i = 0; i < totalPixels; i++) //go to each pixel
    {
        float *distances = (float *)malloc(totalPixels * sizeof(float));
        normalFactor=0;
        for (int j = 0; j < totalPixels; j++)
        {
            dist = calculateGaussianDistance(patches[i], patches[j], patchSize, gaussianWeights); //calculate distances from each patch with gaussian weights
            //printf("dist is %f\n",dist);
            distances[j] = exp(-dist / (sigmaDist * sigmaDist));
            //printf("distances is %f\n",distances[j]);
            normalFactor += distances[j]; //calculate factor to normalize distances ~ Z[i]
        }
        //printf("Normal is %f\n", normalFactor);

        for (int j = 0; j < totalPixels; j++)
            distances[j] /= normalFactor; //distances represents the weight factor for each pixel ~ w(i,j)

        denoisedImage[i] = 0;
        for (int j = 0; j < totalPixels; j++)
            denoisedImage[i] += distances[j] * image[j];
        printf("denoised is %f\n",denoisedImage[i]);
        free(distances);
    }

    printf("Finished denoising \n");
    for (int i = 0; i < totalPixels; i++)
        free(patches[i]);
    free(patches);

    free(gaussianWeights);
    return denoisedImage;
}

float *findRemoved(float *noisy, float *denoised, int size)
{
    int totalPixels = size * size;
    float *removed = (float *)malloc(totalPixels * sizeof(float));
    for (int i = 0; i < totalPixels; i++)
        removed[i] = denoised[i] - noisy[i];
    printf("Finished finding removed\n");
    return removed;
}