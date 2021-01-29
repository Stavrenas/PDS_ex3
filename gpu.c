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


int main(int argc, char *argv[])
{
    int size; //size represents the dimentsions of a **SQUARE** image in pixels //
    char *file = (char *)malloc(20 * sizeof(char));
    char *name = (char *)malloc(20 * sizeof(char));
    sprintf(name, "%s","image3" ) ;
    sprintf(file, "%s.csv", name);
    int *image = readCSV(&size, file);
    printf("Image dimensions are %dx%d\n", size,size);

    double *normal = normalizeImage(image, size); //each pixel has max value of 1 //
    char *normalName = (char *)malloc(20 * sizeof(char));
    sprintf(normalName, "%s_normal", name);
    writeToCSV(normal, size, normalName);

    double *noisy = addNoiseToImage(normal, size); //add gaussian noise to the image//
    char *noisyName = (char *)malloc(20 * sizeof(char));
    sprintf(noisyName, "%s_noisy", name);
    writeToCSV(noisy, size, noisyName);
    foo(size, normal, noisy);

}