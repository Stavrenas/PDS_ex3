#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // sqrt, M_PI
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include "utilities.h"
#include "cudaUtilities.h"

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

float gaussian(float sigma, float x)
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
        array = (int *)realloc(array, pixels * sizeof(int));
    }
    *n = sqrt(pixels);

    fclose(matFile);
    return array;
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
        noisy[i] = (effect + 1) * image[i]; //add gaussian noise
        if (noisy[i] < 0)
            noisy[i] = 0;
        else if (noisy[i] > 1)
            noisy[i] = 1;
    }
    printf("Finished adding noise\n");
    return noisy;
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
            fprintf(filepointer, "%f,", image[i * size + j]); //write each pixel value

        fprintf(filepointer, "\n");
    }
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

float *createPatchesRowMajor(float *image, int size, int patchSize)
{
    //We assume that patchSize is an odd number//
    //In order to create the patches we must consider that the pixels are stored in Row-Major format//
    //A simple aproach is to handle the patches also in the same format//
    int patchLimit = (patchSize - 1) / 2;
    int patchIterator, imageIterator;
    float *patches = (float *)malloc(size * size * patchSize * patchSize * sizeof(float));

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
            for (int o = 0; o < patchSize * patchSize; o++)
                patches[i*size*patchSize*patchSize + j*patchSize*patchSize + o] = patch[o];
            free(patch);
        }
    }
    return patches;
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

float *findRemoved(float *noisy, float *denoised, int size)
{
    int totalPixels = size * size;
    float *removed = (float *)malloc(totalPixels * sizeof(float));
    for (int i = 0; i < totalPixels; i++)
        removed[i] = denoised[i] - noisy[i];
    printf("Finished finding removed\n");
    return removed;
}

// __device__ float getPatchElement(float *image, int size, int pixel, int position, int patchSize)
// {
//     //returns the element in a certain patch position
//     //without a need to save patch in memory

//     int patchLimit = (patchSize - 1) / 2;
//     float result = -1;
//     int j = pixel % size;                                                                    //int i = pixel / size;
//     int m = position % patchSize - patchLimit;                                               //int k = position / patchSize - patchLimit;
//     int imageIterator = (pixel / size + position / patchSize - patchLimit) * size + (j + m); //int imageIterator = (i + k) * size + (j + m);

//     if (imageIterator >= 0 && imageIterator < size * size) //filter out of image pixels
//     {
//         if (!(j < patchLimit && m < -j) && !(j >= size - patchLimit && m >= size - j))
//             result = image[imageIterator];
//     }
//     return result;
// }

// __global__ void distanceSquaredCuda(int size, float *x, float *y, float *z)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < size)
//         z[i] = (x[i] - y[i]) * (x[i] - y[i]);
// }

// __global__ void gaussianDistanceCuda(int size, float *distances, float *gaussianWeights, int patchSize, float *x)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < size)
//     {
//         int patchLimit = (patchSize - 1) / 2;
//         int m = i % patchSize - patchLimit;
//         int k = i / patchSize - patchLimit;
//         int distance = m * m + k * k;
//         x[i] *= gaussianWeights[distance];
//     }
// }


float *calculateDistances(float *patches, int size, int patchSize, float *gaussianWeights, float sigmaDist)
{
    int totalPixels = size * size;
    int patchLimit = (patchSize - 1) / 2;
    float *cudaPatches, *cudaGaussianWeights, *cudaDistances;
    cudaMalloc(&cudaPatches, totalPixels * patchSize * patchSize* sizeof(float));
    cudaMalloc(&cudaDistances, totalPixels * totalPixels * sizeof(float));
    cudaMalloc(&cudaGaussianWeights, (patchSize + patchLimit) * sizeof(float));
    cudaMemcpy(cudaGaussianWeights, gaussianWeights, (patchSize + patchLimit) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaPatches, patches, totalPixels * patchSize * patchSize * sizeof(float), cudaMemcpyHostToDevice);

    findPatchDistances<<<size, size>>>(cudaPatches, size, patchSize, cudaGaussianWeights, cudaDistances, sigmaDist);

    float *distances = (float *)malloc(totalPixels * totalPixels * sizeof(float));
    cudaMemcpy(distances, cudaDistances, (totalPixels * totalPixels * sizeof(float)), cudaMemcpyDeviceToHost);
    cudaFree(cudaPatches);
    cudaFree(cudaDistances);
    cudaFree(cudaGaussianWeights);
    return distances;
}

__global__ void findPatchDistances(float *patches, int size, int patchSize, float *gaussianWeights, float *distances, float sigmaDist)
{

    int col = threadIdx.x; //0.. size-1
    int row = blockIdx.x;  //0.. size-1
    int totalPixels = size * size;
    int totalPatch=patchSize*patchSize;
    //printf("i is %d j is %d\n",i,j);

    for (int i = row*size; i < (row+1)*size; i++)
    {
        for (int j = col*size; j < (col+1)*size; j++)
        {
            int patchLimit = (patchSize - 1) / 2;
            float result, sum = 0;

            for (int k = -patchLimit; k <= patchLimit; k++)
            {
                for (int m = -patchLimit; m <= patchLimit; m++) //go to each pixel of the patch: i*size +j
                {
                    int patchIterator = (k + patchLimit) * patchSize + (m + patchLimit);
                    if (patches[i * totalPatch + patchIterator] != -1 && patches[j * totalPatch + patchIterator] != -1) //this means out of bounds
                    {
                        int distance = m * m + k * k; 
                        result = (patches[i * totalPatch + patchIterator] - patches[j * totalPatch + patchIterator]) *
                            (patches[i * totalPatch + patchIterator] - patches[j * totalPatch + patchIterator]) * gaussianWeights[distance];
                        sum += result;
                    }
                }
            }
            distances[i*totalPixels + j] = exp(-sum / (sigmaDist * sigmaDist));
        }
    }
}
