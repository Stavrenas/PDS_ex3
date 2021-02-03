#ifndef CUDAUTILITIES_H
#define CUDAUTILITIES_H

__device__ float getPatchElement(float *image, int size, int pixel, int position, int patchSize);

__global__ void distanceSquaredCuda(int size, float *x, float *y, float * z);

__global__ void gaussianDistanceCuda(int size, float *distances, float *gaussianWeights, int patchSize, float *x);

float calculateGaussianDistance(float *patch1, float *patch2, int patchSize, float * gaussianWeights);

float *denoise(float *patches, int size, int patchSize, float *gaussianWeights, float sigmaDist, float* image);

float *createPatchesRowMajor(float *image, int size, int patchSize);

__global__ void findPatchDistances(float *patches, int size, int patchSize, float *gaussianWeights, float *distances, float sigmaDist);

__global__ void normalizeDistances(float *distances, int size);

__global__ void calculateDenoisedImage(float *denoised, float *distances, float *image, int size);


#endif