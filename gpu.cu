#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // sqrt, M_PI
#include <stdbool.h>
#include "cudaUtilities.h"
#include "utilities.h"
#include <time.h>
#include <sys/time.h> //gettimeofday

float *denoiseImageCuda(float *image, int size, int patchSize, float sigmaDist, float sigmaGauss);

int main(int argc, char *argv[])
{
    //** GPU-CUDA IMPLEMENTATION **//

    int size; //size represents the dimentsions of a **SQUARE** image in pixels //
    char *file = (char *)malloc(20 * sizeof(char));
    char *name = (char *)malloc(20 * sizeof(char));
    int patchSize;
    float sigmaDist = 0.05;
    float sigmaGauss = 1.66;
    if (argc == 1)
    {
        patchSize = 7;
        sprintf(name, "%s", "image1");
    }
    else if (argc == 3)
    {
        patchSize = atoi(argv[1]);
        sprintf(name, "%s", argv[2]);
    }
    else
    {
        printf("Command line arguements are: PatchSize, name.\n");
        return -1;
    }

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

    struct timeval tStart;
    tStart = tic();
    float *denoised = denoiseImageCuda(noisy, size, patchSize, sigmaDist, sigmaGauss); //remove noise from the image//
    char *denoisedName = (char *)malloc(20 * sizeof(char));
    sprintf(denoisedName, "%s_denoisedCUDA", name);
    writeToCSV(denoised, size, denoisedName);

    printf("%dx%d image with patch size = %d denoised with Cuda in %.6f sec.\n", size, size, patchSize, toc(tStart));

    float *removed = findRemoved(noisy, denoised, size); //find difference from initial image//
    char *removedName = (char *)malloc(20 * sizeof(char));
    sprintf(removedName, "%s_removedCUDA", name);
    writeToCSV(removed, size, removedName);
}

float *denoiseImageCuda(float *image, int size, int patchSize, float sigmaDist, float sigmaGauss)
{
    int patchLimit = (patchSize - 1) / 2;
    float *patches, *denoisedImage, *gaussianWeights;

    gaussianWeights = (float *)malloc((2 * patchLimit * patchLimit + 1) * sizeof(float));
    for (int i = 0; i <= 2 * patchLimit * patchLimit; i++)
        gaussianWeights[i] = gaussian(sigmaGauss, sqrt(i));  //calculate all the necessary gaussian weights//
    patches = createPatchesRowMajor(image, size, patchSize); //patch creation//
    denoisedImage = denoise(patches, size, patchSize, gaussianWeights, sigmaDist, image);

    printf("Finished denoising \n");

    free(patches);
    free(gaussianWeights);
    return denoisedImage;
}
