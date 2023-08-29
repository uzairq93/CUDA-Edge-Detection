#ifndef KERNELS_H_
#define KERNELS_H_

unsigned char* brighten_image(unsigned char* img, int width, int height, int channels, int brightness);
unsigned char* grayscale_image(unsigned char* img, int width, int height, int channels);
unsigned char* cudaErode(unsigned char* img, int width, int height);
unsigned char* cudaDilate(unsigned char* img, int width, int height);
unsigned char* cudaThreshold(unsigned char* img, int width, int height);
unsigned char* cudaGauss(unsigned char* img, int width, int height, int size, float sigma);
unsigned char* cudaSobel(unsigned char* img, int width, int height);
#endif