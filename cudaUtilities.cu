#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // sqrt, M_PI
#include <stdbool.h>
#include <time.h>
#include <sys/time.h> // tic() and toc()
#include "cudaUtilities.h"
#define MAX_SHARED_SIZE 49152

// START OF AUXILIARY FUNCTIONS //
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
            //printf("Finished reading image \n");
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
        noisy[i] = (effect * 0.5 + 1) * image[i]; //add gaussian noise
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
    int patchLimit = (patchSize - 1) / 2;
    int totalPatchSize = patchSize * patchSize;
    int patchIterator, imageIterator;
    float *patches = (float *)malloc(size * size * totalPatchSize * sizeof(float));

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++) //go to each pixel of the image
        {
            float *patch = (float *)malloc(totalPatchSize * sizeof(float)); //We assume that (i,j) is the pixel on the centre
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
            for (int o = 0; o < totalPatchSize; o++)
                patches[i * size * totalPatchSize + j * totalPatchSize + o] = patch[o];
            free(patch);
        }
    }
    return patches;
}

void printPatchRowMajor(float *patches, int patchSize, int i)
{
    int patchI = i * patchSize * patchSize;
    for (int j = patchI; j < patchI + patchSize * patchSize; j++)
    {
        if ((j) % patchSize == 0)
            printf("\n");
        if (patches[j] == -1)
            printf("    x    ");
        else
            printf("%f ", patches[j]);
    }
    printf("\n");
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

// END OF AUXILIARY FUNCTIONS //


float *denoise(float *patches, int size, int patchSize, float *gaussianWeights, float sigmaDist, float *image)
{
    int totalPixels = size * size;
    int patchLimit = (patchSize - 1) / 2;
    float *cudaPatches, *cudaGaussianWeights, *cudaDistances, *cudaDenoised, *cudaImage;

    cudaMalloc(&cudaPatches, totalPixels * patchSize * patchSize * sizeof(float));
    cudaMalloc(&cudaDistances, totalPixels * sizeof(float));
    cudaMalloc(&cudaGaussianWeights, (2*patchLimit*patchLimit+1) * sizeof(float)); //alocate memory for the arrays
    cudaMalloc(&cudaDenoised, totalPixels * sizeof(float));
    cudaMalloc(&cudaImage, totalPixels * sizeof(float));

    cudaMemcpy(cudaGaussianWeights, gaussianWeights, (2*patchLimit*patchLimit+1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaPatches, patches, totalPixels * patchSize * patchSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaImage, image, totalPixels * sizeof(float), cudaMemcpyHostToDevice);

    denoiseKernelShared<<<size, size, MAX_SHARED_SIZE>>>(cudaPatches, size, patchSize, cudaGaussianWeights, sigmaDist, cudaDenoised, cudaImage, cudaDistances);
    float *denoised = (float *)malloc(totalPixels * sizeof(float));
    cudaMemcpy(denoised, cudaDenoised, (totalPixels * sizeof(float)), cudaMemcpyDeviceToHost);

    cudaFree(cudaPatches);
    cudaFree(cudaDistances);
    cudaFree(cudaGaussianWeights);
    cudaFree(cudaDenoised);
    cudaFree(cudaImage);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("Kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    return denoised;
}

__global__ void denoiseKernel(float *patches, int size, int patchSize, float *gaussianWeights, float sigmaDist, float *denoisedImage, float *image, float *distances)
{
    int patchLimit = (patchSize - 1) / 2;
    int gaussianSize = 2 * patchLimit * patchLimit + 1;

    extern __shared__ float shared[];
    int col = threadIdx.x; //0.. size-1 
    int row = blockIdx.x;  //0.. size-1 
    int pixel = row * size + col;
    float pixelValue=0;
    //printf("launched %d block and %d thread\n", blockIdx.x, threadIdx.x);
    float *gaussianWeightsShared=shared;
    // float *patchRowShared=(float *)&gaussianWeightsShared[gaussianSize];
    // for(int i=0; i< totalPatchSize * size; i++)
    //     patchRowShared[i]=patches[i];
    

    for(int g = 0; g < gaussianSize; g++)
        gaussianWeightsShared[g]=gaussianWeights[g];   //load gaussian weights array to shared memory

    float dist,normalFactor = 0.0;
    //float dist =  patchDistanceShared( patchSize, patchShared, &patches[(pixel) * totalPatchSize], gaussianWeightsShared );
    
    //go to each pixel of the image
    //each thread on the same block is on the same j, so all the threads access the same row of the image 
    for (int j = 0; j < size * size; j++){

        //printf("%d thread with j %d and block %d\n", threadIdx.x, j, blockIdx.x);
        dist =  patchDistance( pixel, j, patchSize, patches, gaussianWeightsShared );
        dist = exp(-dist / (sigmaDist * sigmaDist));
        pixelValue += dist * image[j];
        normalFactor += dist;
    }

    denoisedImage[pixel] = pixelValue/normalFactor;//distances now represents the weight factor for each pixel ~ w(i,j)
    
}

__global__ void denoiseKernelShared(float *patches, int size, int patchSize, float *gaussianWeights, float sigmaDist, float *denoisedImage, float *image, float *distances)
{
    int patchLimit = (patchSize - 1) / 2;
    int gaussianSize = 2 * patchLimit * patchLimit + 1;
    int totalPatchSize = patchSize * patchSize;

    extern __shared__ float shared[];
    int col = threadIdx.x; //0.. size-1 
    int row = blockIdx.x;  //0.. size-1 
    int pixel = row * size + col;
    float pixelValue=0;
    //printf("launched %d block and %d thread\n", blockIdx.x, threadIdx.x);
    float *gaussianWeightsShared=shared;
    float *patchesRowShared=(float *)&gaussianWeightsShared[gaussianSize];
    float *patchShared=(float *)&patchesRowShared[totalPatchSize * size];


    for(int g = 0; g < gaussianSize; g++)
        gaussianWeightsShared[g]=gaussianWeights[g];   //load gaussian weights array to shared memory

    for(int p = 0; p < totalPatchSize; p++)
        patchShared[p]=patches[pixel * totalPatchSize + p]; //load pixel's patch to shared memory

    float dist,normalFactor = 0.0;

    //go to each pixel of the image
    //each thread on the same block is on the same j, so all the threads access the same row of the image 
    for (int j = 0; j < size ; j++){
        printf("%d thread with j %d and block %d\n", threadIdx.x, j, blockIdx.x);
        for(int s = 0; s< totalPatchSize * size; s++)
            patchesRowShared[s]=patches[s]; //load patches of row to shared memory

        for (int i = 0; i < size ; i++){
            printf("%d thread with i %d and block %d\n", threadIdx.x, i, blockIdx.x);
            
            dist =  patchDistanceSingle( patchSize, patchShared,  &patchesRowShared[(i) * totalPatchSize ], gaussianWeightsShared );
            dist = exp(-dist / (sigmaDist * sigmaDist));
            pixelValue += dist * image[j];
            normalFactor += dist;
        }
    }

    denoisedImage[pixel] = pixelValue/normalFactor;//distances now represents the weight factor for each pixel ~ w(i,j)
    
}

__device__ float patchDistance(int i, int j, int patchSize, float *patches, float *gaussianWeights)
{
    float sum = 0;
    int patchLimit = (patchSize - 1) / 2;
    int totalPatchSize = patchSize * patchSize;
    for (int k = -patchLimit; k <= patchLimit; k++)
    {
        for (int m = -patchLimit; m <= patchLimit; m++) //go to each pixel of patch(i) and patch(j)
        {
            int patchIterator = (k + patchLimit) * patchSize + (m + patchLimit);
            if (patches[i * totalPatchSize + patchIterator] != -1 && patches[j * totalPatchSize + patchIterator] != -1) //this means out of bounds
            {
                int distance = m * m + k * k;
                sum += (patches[i * totalPatchSize + patchIterator] - patches[j * totalPatchSize + patchIterator]) *
                       (patches[i * totalPatchSize + patchIterator] - patches[j * totalPatchSize + patchIterator]) * gaussianWeights[distance];
            }
        }
    }
    return sum;
}

__device__ float patchDistanceSingle(int patchSize, float *patch1, float *patch2, float *gaussianWeightsShared)
{
    float sum = 0;
    int patchLimit = (patchSize - 1) / 2;
    for (int k = -patchLimit; k <= patchLimit; k++)
    {
        for (int m = -patchLimit; m <= patchLimit; m++) //go to each pixel of patch(i) and patch(j)
        {
            int patchIterator = (k + patchLimit) * patchSize + (m + patchLimit);
            if (patch1[patchIterator] != -1 && patch2[patchIterator] != -1) //this means out of bounds
            {
                int distance = m * m + k * k;
                sum += (patch1[patchIterator] - patch2[patchIterator]) *
                       (patch1[patchIterator] - patch2[patchIterator]) * gaussianWeightsShared[distance];
            }
        }
    }
    return sum;
}
