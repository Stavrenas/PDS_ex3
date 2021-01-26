#ifndef UTILITIES_H
#define UTILITIES_H

double gaussian(double sigma, double x);

int *readCSV(int *n, char *file);

double *normalizeImage(int *image, int size);

double *addNoiseToImage(double *image, int size);

void writeToCSV(double* image, int size, char* name);

double findMax(double * array, int size );

double **createPatches(double *image, int size, int patchSize);

double calculateGaussianDistance(double *patch1, double *patch2, int patchSize, double *gaussianWeights);

void printPatch(double *patch, int patchSize);

double *denoiseImage(double *image, int size, int patchSize, double sigmaDist, double sigmaGauss);

double * findRemoved(double * noisy, double *denoised, int size);

#endif