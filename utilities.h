#ifndef UTILITIES_H
#define UTILITIES_H

double gaussian(double sigma, double x);

double *gaussianWeight(int patchSize);

bool isPowerOfTwo(int n);

int *readCSV(int *n, char *file);

double *normalizeImage(int *image, int size);

double *addNoiseToImage(double *image, int size);

void writeToCSV(double* image, int size, char* name);

double findMax(double * array, int size );

#endif