#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // sqrt, M_PI
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
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


__device__ double getPatchElement(double *image, int size, int pixel, int position, int patchSize)
{ 
    //returns the element in a certain patch position 
    //without a need to save patch in memory

    int patchLimit = (patchSize - 1) / 2;
    double result = -1;
    int j = pixel % size;                       //int i = pixel / size;
    int m = position % patchSize - patchLimit;  //int k = position / patchSize - patchLimit;
    int imageIterator = (pixel / size + position / patchSize - patchLimit) * size + (j + m); //int imageIterator = (i + k) * size + (j + m);

    if (imageIterator >= 0 && imageIterator < size * size) //filter out of image pixels
    {
        if (!(j < patchLimit && m < -j) && !(j >= size - patchLimit && m >= size - j))
            result = image[imageIterator];
    }
    return result;
}


__global__ void distanceSquared(int size, double *x, double *y, double * z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        z[i] = (x[i] - y[i] )* (x[i] - y[i]);
}


// __global__ void gaussianDistance(int size, double *distances, double *gaussianWeights, int patchSize, double *x){
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < size){
//         int patchLimit = (patchSize-1) / 2;
//         int m = i % patchSize - patchLimit;
//         int k = i / patchSize - patchLimit;
//         int distance= m * m + k * k;
//         x[i]*=gaussianWeights[distance];
//     }
// }

// __global__ double calculateGaussianDistance(double *patch1, double *patch2, int patchSize, double * gaussianWeights)
// {   
//     int patchLimit = (patchSize-1) / 2;
//     int totalPatchPixels=patchSize*patchSize;
//     double * results, *distances, sum;
//     cudaMalloc(&results, (patchSize + patchLimit) * sizeof(double));
//     cudaMalloc(&distances, (totalPatchPixels) * sizeof(double));

//     distanceSquared<<<1,totalPatchPixels>>>(totalPatchPixels, patch1, patch2, distances);
//     gaussianDistance<<<1,totalPatchPixels>>>(totalPatchPixels, distances, gaussianWeights, patchSize, results);
//     sum=0;
//     for(int i=0; i<totalPatchPixels; i++)
//     sum+=results[i];
    
//     return sum;
// }



// __global__ double *denoiseImage(double *image, int size, int patchSize, double sigmaDist, double sigmaGauss)
// {

//     int totalPixels = size * size;
//     int patchLimit = (patchSize - 1) / 2;
//     double **patches = (double **)malloc(totalPixels * sizeof(double *));
//     double **distances = (double **)malloc(totalPixels * sizeof(double *));
//     patches = createPatches(image, size, patchSize);

//     double *gaussianWeights = (double *)malloc((patchSize + patchLimit) * sizeof(double));
//     for (int i = 0; i < patchSize + patchLimit; i++)
//         gaussianWeights[i] = gaussian(sigmaGauss, i);

//     for (int i = 0; i < totalPixels; i++)
//     {
//         double normalFactor = 0;
//         distances[i] = (double *)malloc(totalPixels * sizeof(double));

//         for (int j = 0; j < totalPixels; j++)
//         {
//             double dist = calculateGaussianDistance(patches[i], patches[j], patchSize, gaussianWeights); //calculate distances from each patch with gaussian weights
//             distances[i][j] = exp(-dist / (sigmaDist * sigmaDist));
//             normalFactor += distances[i][j]; //calculate factor to normalize distances ~ Z[i]
//         }

//         for (int j = 0; j < totalPixels; j++)
//             distances[i][j] /= normalFactor; //distances represents the weight factor for each pixel ~ w(i,j)
//     }

//     double *denoisedImage = (double *)malloc(totalPixels * sizeof(double));
//     for (int i = 0; i < totalPixels; i++)
//     {
//         denoisedImage[i] = 0;
//         for (int j = 0; j < totalPixels; j++)
//             denoisedImage[i] += distances[i][j] * image[j];
//     }
//     printf("Finished denoising \n");
//     for (int i = 0; i < totalPixels; i++)
//     {
//         free(patches[i]);
//         free(distances[i]);
//     }
//     free(patches);
//     free(distances);
//     return denoisedImage;
// }


extern "C" void foo(int size, double *normal, double* noisy){

    size*=size;
    double * cudapatch1, *cudapatch2,*distances;
    cudaMalloc(&distances, (size) * sizeof(double));
    cudaMalloc(&cudapatch1, size*sizeof(double));
    cudaMalloc(&cudapatch2, size*sizeof(double));
    cudaMemcpy(cudapatch1, normal, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cudapatch2, noisy, size * sizeof(double), cudaMemcpyHostToDevice);
    int iterations=1000000;
    struct timeval tStart;
	tStart = tic();
    for(int j=0; j<iterations; j++)
    distanceSquared<<<1, size>>>(size,cudapatch1, cudapatch2, distances);

    cudaFree(cudapatch1);
    cudaFree(cudapatch2);

    double * hostResults = (double *)malloc(size * sizeof(double));
    cudaMemcpy(hostResults, distances, size * sizeof(double), cudaMemcpyDeviceToHost);


    // for(int i =0 ; i< size; i ++)
    //     printf("%f\n", hostResults[i]);

    printf("cuda took %.6f sec for %d image\n", toc(tStart),size);

    tStart = tic();
    double * dist = (double *)malloc(size * sizeof(double));
    for(int j=0; j<iterations; j++){
    for(int i =0 ; i< size; i ++)
    dist[i]=(normal[i]-noisy[i])*(normal[i]-noisy[i]);
    // for(int i =0 ; i< size; i ++)
    //     printf("%f\n", dist[i]);
    }
    printf("cpu took %.6f sec for %d image\n", toc(tStart),size);
}
