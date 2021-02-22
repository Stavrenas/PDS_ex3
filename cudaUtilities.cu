#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // sqrt, M_PI
#include <stdbool.h>
#include <time.h>
#include <sys/time.h> // tic() and toc()
#include "cudaUtilities.h"
#define MAX_DISTANCE_SIZE 256

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

// START OF CUDA FUNCTIONS **NOT** USING SHARED MEMORY //
float *denoise(float *patches, int size, int patchSize, float *gaussianWeights, float sigmaDist, float *image)
{
    int totalPixels = size * size;
    int patchLimit = (patchSize - 1) / 2;
    float *cudaPatches, *cudaGaussianWeights, *cudaDistances, *denoisedCuda, *imageCuda;

    cudaMalloc(&cudaPatches, totalPixels * patchSize * patchSize * sizeof(float));
    cudaMalloc(&cudaDistances, totalPixels * sizeof(float));
    cudaMalloc(&cudaGaussianWeights, (patchSize + patchLimit) * sizeof(float)); //alocate memory for the arrays
    cudaMalloc(&denoisedCuda, totalPixels * sizeof(float));
    cudaMalloc(&imageCuda, totalPixels * sizeof(float));

    cudaMemcpy(cudaGaussianWeights, gaussianWeights, (patchSize + patchLimit) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaPatches, patches, totalPixels * patchSize * patchSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(imageCuda, image, totalPixels * sizeof(float), cudaMemcpyHostToDevice);

    // UNIFY THESE //

    // findPatchDistances<<<size, size>>>(cudaPatches, size, patchSize, cudaGaussianWeights, cudaDistances, sigmaDist);
    // cudaDeviceSynchronize(); //ensure that cudaPatches is fully calculated

    // normalizeDistances<<<size, size>>>(cudaDistances, size);
    // cudaDeviceSynchronize();

    // calculateDenoisedImage<<<size, size>>>(denoisedCuda, cudaDistances, imageCuda, size);
    // cudaDeviceSynchronize(); //ensure that denoisedCuda is properly calculated

    //            //
    printf("Dist size is %ld bytes or %f Kbytes",totalPixels*sizeof(float),(float)(totalPixels*sizeof(float)/1024));
    denoiseKernel<<<size, size, totalPixels*sizeof(float)>>>(cudaPatches, size, patchSize, cudaGaussianWeights, sigmaDist, denoisedCuda, imageCuda);
    cudaDeviceSynchronize();

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));

    float *denoised = (float *)malloc(totalPixels * sizeof(float));
    cudaMemcpy(denoised, denoisedCuda, (totalPixels * sizeof(float)), cudaMemcpyDeviceToHost);

    cudaFree(cudaPatches);
    cudaFree(cudaDistances);
    cudaFree(cudaGaussianWeights);
    cudaFree(denoisedCuda);
    cudaFree(imageCuda);

    return denoised;
}

// __global__ void findPatchDistances(float *patches, int size, int patchSize, float *gaussianWeights, float *distances, float sigmaDist)
// {

//     int col = threadIdx.x; //0.. size-1
//     int row = blockIdx.x;  //0.. size-1
//     int totalPixels = size * size;
//     //Each thread goes to a single row and column and calculates the distances.
//     //For example,  in the block with blockIdx.x = 50, all the threads find the distances
//     //of each pixel in the image with each pixel of the 50th row
//     for (int i = row * size; i < (row + 1) * size; i++)
//     {
//         for (int j = col * size; j < (col + 1) * size; j++)
//         {
//             float dist = patchDistance(i, j, patchSize, patches, gaussianWeights);
//             distances[i * totalPixels + j] = exp(-dist / (sigmaDist * sigmaDist));
//         }
//     }
// }

// __global__ void normalizeDistances(float *distances, int size)
// {
//     int totalPixels = size * size;
//     float normalFactor = 0;
//     int i = threadIdx.x + blockIdx.x * blockDim.x;

//     for (int j = 0; j < totalPixels; j++)
//         normalFactor += distances[i * totalPixels + j]; //calculate factor to normalize distances ~ Z[i]

//     for (int j = 0; j < totalPixels; j++)
//         distances[i * totalPixels + j] /= normalFactor; //distances represents the weight factor for each pixel ~ w(i,j)
//     //printf("%dth pixel normal factor %f\n", i, normalFactor);
// }

__global__ void denoiseKernel(float *patches, int size, int patchSize, float *gaussianWeights, float sigmaDist, float *denoisedImage, float *image)
{
    extern __shared__ float distances[];
    int col = threadIdx.x; //0.. size-1 -> 
    int row = blockIdx.x;  //0.. size-1 -> Current row of the image
    //int totalPixels = size * size;

    //int i = row * size + col;
    //printf("i is %d \n",i);
    printf("Lanched %d thread in %d block\n",threadIdx.x, blockIdx.x);
    //go to each pixel of the row
    for(int i=row*size ; i< (row+1)*size; i++){

    //go to each pixel of the image
    //j represents current row
    for(int j=0; j<size; j++){
    float dist = patchDistance(i, j*size+col, patchSize, patches, gaussianWeights);
    distances[j*size+col]  = exp(-dist / (sigmaDist * sigmaDist));
    //printf("i is %d, j*size+col is %d, dist is %f, j is %d, col is %d, size is %d\n", i, j*size+col, distances[j*size+col],j,col, size);
    }
    __syncthreads();
    float normalFactor = 0;
    for (int j = 0; j < size*size; j++){

        normalFactor += distances[j]; //calculate factor to normalize distances ~ Z[i]
        //printf ("%d thread: dist is %f\n",threadIdx.x,distances[j]);
    }
    // if(col==0)
    // printf("%dth pixel normal factor %f\n", i, normalFactor);
    
    for (int j = 0; j < size; j++)
        distances[j*size+col] /= normalFactor; //distances represents the weight factor for each pixel ~ w(i,j)
    
    __syncthreads();

    if(threadIdx.x==0){
    denoisedImage[i] = 0;
    for (int j = 0; j < size*size; j++)
        denoisedImage[i] += distances[j] * image[j];
    }
}
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

// __global__ void calculateDenoisedImage(float *denoisedImage, float *distances, float *image, int size)
// {
//     int totalPixels = size * size;
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     denoisedImage[i] = 0;
//     for (int j = 0; j < totalPixels; j++)
//         denoisedImage[i] += distances[i * totalPixels + j] * image[j];
// }

// END OF CUDA FUNCTIONS **NOT** USING SHARED MEMORY //

// START OF CUDA FUNCTIONS USING SHARED MEMORY //

// float *denoiseShared(float *patches, int size, int patchSize, float *gaussianWeights, float sigmaDist, float *image)
// {
//     int totalPixels = size * size;
//     int patchLimit = (patchSize - 1) / 2;
//     float *cudaPatches, *cudaGaussianWeights, *cudaDistances, *denoisedCuda, *imageCuda;

//     cudaMalloc(&cudaPatches, totalPixels * patchSize * patchSize * sizeof(float));
//     cudaMalloc(&cudaDistances, totalPixels * totalPixels * sizeof(float));
//     cudaMalloc(&cudaGaussianWeights, (patchSize + patchLimit) * sizeof(float)); //alocate memory for the arrays
//     cudaMalloc(&denoisedCuda, totalPixels * sizeof(float));
//     cudaMalloc(&imageCuda, totalPixels * sizeof(float));

//     cudaMemcpy(cudaGaussianWeights, gaussianWeights, (patchSize + patchLimit) * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(cudaPatches, patches, totalPixels * patchSize * patchSize * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(imageCuda, image, totalPixels * sizeof(float), cudaMemcpyHostToDevice);

//     findPatchDistancesShared<<<size, size>>>(cudaPatches, size, patchSize, cudaGaussianWeights, cudaDistances, sigmaDist);
//     cudaDeviceSynchronize(); //ensure that cudaPatches is fully calculated

//     normalizeDistancesShared<<<size, size>>>(cudaDistances, size);
//     cudaDeviceSynchronize();

//     calculateDenoisedImageShared<<<size, size>>>(denoisedCuda, cudaDistances, imageCuda, size);
//     cudaDeviceSynchronize(); //ensure that denoisedCuda is properly calculated

//     float *denoised = (float *)malloc(totalPixels * sizeof(float));
//     cudaMemcpy(denoised, denoisedCuda, (totalPixels * sizeof(float)), cudaMemcpyDeviceToHost);

//     cudaFree(cudaPatches);
//     cudaFree(cudaDistances);
//     cudaFree(cudaGaussianWeights);
//     cudaFree(denoisedCuda);
//     cudaFree(imageCuda);

//     return denoised;
// }

// __global__ void findPatchDistancesShared(float *patches, int size, int patchSize, float *gaussianWeights, float *distances, float sigmaDist)
// {

//     int col = threadIdx.x; //0.. size-1
//     int row = blockIdx.x;  //0.. size-1
//     int patchLimit = (patchSize - 1) / 2;
//     int totalPixels = size * size;
//     int totalPatchSize = patchSize * patchSize;
//     if (col == 0)
//         printf("%d \n", row);
//     //Each thread goes to a single row and column and calculates the distances.
//     //For example,  in the block with blockIdx.x = 50, all the threads find the distances
//     //of each pixel in the image with each pixel of the 50th row
//     for (int i = row * size; i < (row + 1) * size; i++)
//     {
//         for (int j = col * size; j < (col + 1) * size; j++)
//         {
//             float sum = 0;
//             for (int k = -patchLimit; k <= patchLimit; k++)
//             {
//                 for (int m = -patchLimit; m <= patchLimit; m++) //go to each pixel of patch(i) and patch(j)
//                 {
//                     int patchIterator = (k + patchLimit) * patchSize + (m + patchLimit);
//                     if (patches[i * totalPatchSize + patchIterator] != -1 && patches[j * totalPatchSize + patchIterator] != -1) //this means out of bounds
//                     {
//                         int distance = m * m + k * k;
//                         sum += (patches[i * totalPatchSize + patchIterator] - patches[j * totalPatchSize + patchIterator]) *
//                                (patches[i * totalPatchSize + patchIterator] - patches[j * totalPatchSize + patchIterator]) * gaussianWeights[distance];
//                     }
//                 }
//             }
//             distances[i * totalPixels + j] = exp(-sum / (sigmaDist * sigmaDist));
//         }
//     }
// }

// __global__ void normalizeDistancesShared(float *distances, int size)
// {
//     int totalPixels = size * size;
//     float normalFactor = 0;
//     int i = threadIdx.x + blockIdx.x * blockDim.x;

//     for (int j = 0; j < totalPixels; j++)
//         normalFactor += distances[i * totalPixels + j]; //calculate factor to normalize distances ~ Z[i]

//     for (int j = 0; j < totalPixels; j++)
//         distances[i * totalPixels + j] /= normalFactor; //distances represents the weight factor for each pixel ~ w(i,j)
//     //printf("%dth pixel normal factor %f\n", i, normalFactor);
// }

// __global__ void calculateDenoisedImageShared(float *denoisedImage, float *distances, float *image, int size)
// {
//     int totalPixels = size * size;
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     denoisedImage[i] = 0;
//     for (int j = 0; j < totalPixels; j++)
//         denoisedImage[i] += distances[i * totalPixels + j] * image[j];
// }
// END OF CUDA FUNCTIONS  USING SHARED MEMORY //