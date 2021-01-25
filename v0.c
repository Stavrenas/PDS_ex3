#include <stdio.h>
#include <stdlib.h>
#include <math.h> // sqrt, M_PI
#include <stdbool.h>
#include "utilities.h"
#include <time.h>
#include <sys/time.h>

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

/* Usage of the functions above 

        struct timeval tStart;
		tStart = tic();
		
        run_function;
		
		printf("%.6f\n", toc(tStart));


*/

double *denoiseImage(double *image, int size, int patchSize, double sigma);

int main(int argc, char **argv[])
{
    // int patchSize = 5;
    // double *weights = (double *)malloc(patchSize * sizeof(double));
    // weights = gaussianWeight(patchSize);
    // for (int i = 0; i < patchSize; i++)
    //     printf("weight no %d is %f\n", i, weights[i]);

    //** CPU IMPLEMENTATION **//

    int n; //n represents the dimentsions of a **SQUARE** image in pixels //
    char *file = (char *)malloc(20 * sizeof(char));
    char *name = (char *)malloc(20 * sizeof(char));
    name = "image3";
    sprintf(file, "%s.csv", name);
    int *image = readCSV(&n, file);
    printf("size is %d\n", n);

    double *normal = normalizeImage(image, n); //each pixel has max value of 1 //
    char *normalName = (char *)malloc(20 * sizeof(char));
    sprintf(normalName, "%s_normal", name);
    writeToCSV(normal, n, normalName);

    double *noisy = addNoiseToImage(normal, n); //add gaussian noise to the image//
    char *noisyName = (char *)malloc(20 * sizeof(char));
    sprintf(noisyName, "%s_noisy", name);
    writeToCSV(noisy, n, noisyName);
}

double *denoiseImage(double *image, int size, int patchSize, double sigma)
{
    //We assume that patchSize is an odd number//
    //In order to create the patches we must consider that the pixels are stored in Row-Major format//
    //A simple aproach is to handle the patches also in the same format//
    int patchLimit = patchSize / 2;
    int patchIterator, imageIterator;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++) //go to each pixel of the image
        {
            int mod = j % size;
            double *patch = (double *)malloc(patchSize * sizeof(double)); //We assume that (i,j) is the pixel on the centre
            for (int k = -patchLimit; k <= patchLimit; k++)
            {
                for (int m = -patchLimit; m <= patchLimit; m++) //go to each pixel of the patch: i*size +j
                {
                    patchIterator = (k + patchLimit) * patchSize + (m + patchLimit);
                    imageIterator = (i + k) * size + (j + m);

                    if (imageIterator >= 0 && imageIterator < size * size) //filter out of image pixels
                    {

                        if (!(mod < patchLimit && m < -mod) && !(mod >= size - patchLimit && m >= size - mod))
                            //!(j % size < patchLimit && m +  < 0) filters pixels that are on the left side of the patch
                            //!(j % size >= size - patchLimit && m  >=size - j % size) filters pixels that are on the right side of the patch

                            printf("patch: %d , image: %d, m is %d, size - mod is %d\n", patchIterator, imageIterator, m, size - mod);
                    }

                    // patch[patchIterator] = -1;
                    // patch[patchIterator] = image [imageIterator];
                }
            }
            printf("\n");
        }
    }
}