#ifndef CUDAUTILITIES_H
#define CUDAUTILITIES_H

float *createPatchesRowMajor(float *image, int size, int patchSize);

float *denoise(float *patches, int size, int patchSize, float *gaussianWeights, float sigmaDist, float* image);

__global__ void denoiseKernel(float *patches, int size, int patchSize, float *gaussianWeights, float sigmaDist, float *denoisedImage, float *image, float *distances);

__device__ float patchDistance(int i, int j,int patchSize, float *patches, float *gaussianWeights );

__global__ void denoiseKernelShared(float *patches, int size, int patchSize, float *gaussianWeights, float sigmaDist, float *denoisedImage, float *image, float *distances);

__device__ float patchDistanceSingle(int patchSize, float *patch1, float *patch2, float *gaussianWeightsShared);

#endif
