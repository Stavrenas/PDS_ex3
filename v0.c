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

int main(int argc, char **argv[])
{
    // int patchSize = 5;
    // double *weights = (double *)malloc(patchSize * sizeof(double));
    // weights = gaussianWeight(patchSize);
    // for (int i = 0; i < patchSize; i++)
    //     printf("weight no %d is %f\n", i, weights[i]);

    //** CPU IMPLEMENTATION **//

    int size; //size represents the dimentsions of a **SQUARE** image in pixels //
    char *file = (char *)malloc(20 * sizeof(char));
    char *name = (char *)malloc(20 * sizeof(char));
    name = "image3";
    sprintf(file, "%s.csv", name);
    int *image = readCSV(&size, file);
    printf("size is %d\n", size);

    double *normal = normalizeImage(image, size); //each pixel has max value of 1 //
    char *normalName = (char *)malloc(20 * sizeof(char));
    sprintf(normalName, "%s_normal", name);
    writeToCSV(normal, size, normalName);

    double *noisy = addNoiseToImage(normal, size); //add gaussian noise to the image//
    char *noisyName = (char *)malloc(20 * sizeof(char));
    sprintf(noisyName, "%s_noisy", name);
    writeToCSV(noisy, size, noisyName);

    int patchSize = 3;
    double sigmaDist = 0.5; //for 3x3 best is 0.5
    double sigmaGauss = 1.66;
    struct timeval tStart;
	tStart = tic();
    double *denoised = denoiseImage(noisy, size, patchSize, sigmaDist, sigmaGauss); //remove noise from the image//
    char *denoisedName = (char *)malloc(20 * sizeof(char));
    sprintf(denoisedName, "%s_denoised", name);
    writeToCSV(denoised, size, denoisedName);
    printf("Took %.6f sec for %dx%d image with patch size = %d\n", toc(tStart),size,size,patchSize);

    double *removed = findRemoved(noisy, denoised,size); //find difference//
    char *removedName = (char *)malloc(20 * sizeof(char));
    sprintf(removedName, "%s_removed", name);
    writeToCSV(removed, size, removedName);


}


