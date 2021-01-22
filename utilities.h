#ifndef UTILITIES_H
#define UTILITIES_H

double gaussian(double sigma, double x);

double *gaussianWeight(int patchSize);

bool isPowerOfTwo(int n);

int *readCSV(int *n, char *file);

#endif 