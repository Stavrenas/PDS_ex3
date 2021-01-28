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
            printf("Finished reading image \n");
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

double *normalizeImage(int *image, int size) //size represents the dimension
{ 
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
        noisy[i] = (effect + 1) * image[i]; //add gaussian noise
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
            fprintf(filepointer, "%f,", image[i * size + j]); //write each pixel value

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

                        if (!(j < patchLimit && m < -j) && !(j >= size - patchLimit && m >= size - j))
                            //!(j % size < patchLimit && m +  < 0) filters pixels that are on the left side of the patch
                            //!(j % size >= size - patchLimit && m  >=size - j % size) filters pixels that are on the right side of the patch
                            patch[patchIterator] = image[imageIterator];
                    }
                }
            }
            patches[i * size + j] = patch;
        }
    }
    return patches;
}

double calculateGaussianDistance(double *patch1, double *patch2, int patchSize, double * gaussianWeights)
{
    int patchLimit = (patchSize-1) / 2;
    double sum, result, gauss = 0;
    sum = 0;

    for (int k = -patchLimit; k <= patchLimit; k++)
    {
        for (int m = -patchLimit; m <= patchLimit; m++) //go to each pixel of the patch: i*size +j
        {
            int patchIterator = (k + patchLimit) * patchSize + (m + patchLimit);
            if (patch1[patchIterator] != -1 && patch2[patchIterator] != -1) //this means out of bounds
            {
                int distance = m * m + k * k; //distance from centre pixel
                result = (patch1[patchIterator] - patch2[patchIterator]) * (patch1[patchIterator] - patch2[patchIterator]) * gaussianWeights[distance];
                sum += result;
            }
        }
    }
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
    int patchLimit = (patchSize-1) / 2;
    double **patches = (double **)malloc(totalPixels * sizeof(double *));
    double **distances = (double **)malloc(totalPixels * sizeof(double *));
    patches = createPatches(image, size, patchSize);

    double *gaussianWeights = (double *)malloc((patchSize + patchLimit) * sizeof(double));
    for (int i = 0; i < patchSize + patchLimit; i++)
        gaussianWeights[i] = gaussian(sigmaGauss, i);

    for (int i = 0; i < totalPixels; i++)
    {
        double normalFactor = 0;
        distances[i] = (double *)malloc(totalPixels * sizeof(double));

        for (int j = 0; j < totalPixels; j++)
        {
            double dist = calculateGaussianDistance(patches[i], patches[j], patchSize, gaussianWeights); //calculate distances from each patch with gaussian weights
            distances[i][j] = exp(-dist / (sigmaDist * sigmaDist));
            normalFactor += distances[i][j]; //calculate factor to normalize distances ~ Z[i]
        }

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