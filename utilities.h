#ifndef UTILITIES_H
#define UTILITIES_H
#include <sys/time.h>

float gaussian(float sigma, float x);

int *readCSV(int *n, char *file);

float *normalizeImage(int *image, int size);

float *addNoiseToImage(float *image, int size);

void writeToCSV(float* image, int size, char* name);

float findMax(float * array, int size );

float **createPatches(float *image, int size, int patchSize);

float calculateGaussianDistance(float *patch1, float *patch2, int patchSize, float *gaussianWeights);

void printPatch(float *patch, int patchSize);

float *denoiseImage(float *image, int size, int patchSize, float sigmaDist, float sigmaGauss);

float * findRemoved(float * noisy, float *denoised, int size);

double toc(struct timeval begin);

struct timeval tic();

#endif