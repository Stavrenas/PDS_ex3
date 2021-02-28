Code usage;

First , make sure to;
make gpu
make gpu


Both implementations have the following input arguements;
1) an integer corresponding to the patch size 
2) the name of the image file in csv format,  wihtout the ".csv" extension

For example , if an image is stored in house.csv file, for a patch size of 5 we do; ./gpu 5 house

After completion, the files house_normal.csv, house_noisy.csv, house_denoised.csv and house_removed.csv 

hose_normal.csv is the normalized image, meaning the maximum pixel value is 1
house_noisy.csv is the same image with gaussian noise implemented
house_denoised.csv is the denoised image using the non local means algorithm
house_removed.csv visualizes noise reduction

In order to visualize the images, the file show_imagesCPU.m (for the CPU implementation) or show_imagesGPU.m (for the CUDA implementation) can be executed, using MATLAB or octave . The user must change
the string named start with the name of the initial image file, wihtout the ".csv" extension.
