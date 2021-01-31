#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // sqrt, M_PI
#include <stdbool.h>
#include "cudaUtilities.h"
#include "utilities.h"
#include <time.h>
#include <sys/time.h>

float *denoiseImageCuda(float *image, int size, int patchSize, float sigmaDist, float sigmaGauss);

int main(int argc, char *argv[])
{
    //** GPU-CUDA IMPLEMENTATION **//

    int size; //size represents the dimentsions of a **SQUARE** image in pixels //
    char *file = (char *)malloc(20 * sizeof(char));
    char *name = (char *)malloc(20 * sizeof(char));
    sprintf(name, "%s", "image2");
    sprintf(file, "%s.csv", name);
    int *image = readCSV(&size, file);
    printf("Image dimensions are %dx%d\n", size, size);

    float *normal = normalizeImage(image, size); //each pixel has max value of 1 //
    char *normalName = (char *)malloc(20 * sizeof(char));
    sprintf(normalName, "%s_normal", name);
    writeToCSV(normal, size, normalName);

    float *noisy = addNoiseToImage(normal, size); //add gaussian noise to the image//
    char *noisyName = (char *)malloc(20 * sizeof(char));
    sprintf(noisyName, "%s_noisy", name);
    writeToCSV(noisy, size, noisyName);

    int patchSize = 3;
    float sigmaDist = 0.05;
    float sigmaGauss = 1.66;

    // float noisy[49] = {0.12, 0.56, 0.34, 0.11, 1, 0, 0.93,
    //                     0.77, 0.33, 0.11, 0.24, 0.9, 0.34, 0.11,
    //                     0.32, 0.35, 0.48, 0.69, 0.21, 0, 0.45,
    //                     0.98, 0.45, 0.69, 0.54, 0.26, 0.45, 0.23,
    //                     0.69, 0.21, 0, 0.77, 0.33, 0.11, 1,
    //                     0.9, 0.34, 0.11, 0.45, 0.69, 0.54, 0.72,
    //                     0.11, 0.24, 0.9, 0.34, 0.11, 0.11, 0.32};
    // size = 7;
    // patchSize = 3;

    struct timeval tStart;
    tStart = tic();

    float *denoised = denoiseImageCuda(noisy, size, patchSize, sigmaDist, sigmaGauss); //remove noise from the image//
    char *denoisedName = (char *)malloc(20 * sizeof(char));
    sprintf(denoisedName, "%s_denoisedCUDA", name);
    writeToCSV(denoised, size, denoisedName);

    printf("Cuda took %.6f sec for %dx%d image with patch size = %d\n", toc(tStart), size, size, patchSize);

    float *removed = findRemoved(noisy, denoised, size); //find difference from original//
    char *removedName = (char *)malloc(20 * sizeof(char));
    sprintf(removedName, "%s_removedCUDA", name);
    writeToCSV(removed, size, removedName);
}

float *denoiseImageCuda(float *image, int size, int patchSize, float sigmaDist, float sigmaGauss)
{
    int totalPixels = size * size;
    int patchLimit = (patchSize - 1) / 2;
    float *patches ,*distances; 

    float *gaussianWeights = (float *)malloc((patchSize + patchLimit) * sizeof(float));
    for (int i = 0; i < patchSize + patchLimit; i++)
        gaussianWeights[i] = gaussian(sigmaGauss, i); //calculate all the necessary gaussian weights
    struct timeval tStart;
    tStart = tic();
    patches = createPatchesRowMajor(image, size, patchSize); //patch creation
    distances = calculateDistances(patches, size, patchSize, gaussianWeights, sigmaDist);
    printf("gpu time  %.6f sec \n", toc(tStart));
    tStart = tic();

    //TODO: CUDA THESE
    for (int i = 0; i < totalPixels; i++)
    {
        float normalFactor = 0;
        for (int j = 0; j < totalPixels; j++)
            normalFactor += distances[i * totalPixels + j]; //calculate factor to normalize distances ~ Z[i]

        for (int j = 0; j < totalPixels; j++)
            distances[i * totalPixels + j] /= normalFactor; //distances represents the weight factor for each pixel ~ w(i,j)
    }

    float *denoisedImage = (float *)malloc(totalPixels * sizeof(float));
    for (int i = 0; i < totalPixels; i++)
    {
        denoisedImage[i] =0;
        for (int j = 0; j < totalPixels; j++)
            denoisedImage[i]+=distances[i * totalPixels + j] * image[j];

    }
    //
    printf("Cpu time  %.6f sec \n", toc(tStart));

    printf("Finished denoising \n");

    free(patches);
    free(distances);
    free(gaussianWeights);
    return denoisedImage;
}