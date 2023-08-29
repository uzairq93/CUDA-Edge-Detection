#include <iostream>
#include <opencv2/opencv.hpp>
#include "kernel.h"
#include <cmath>

using namespace std;
using namespace cv;

#include <chrono>
using namespace std::chrono;

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#define FILTERSIZE 15
#define SIGMA 4.0f
#define GREYCHAN 2


float** kernel_gen(int size, float sigma)
{
    float** kernel = (float **)malloc(size * sizeof(float *));
     for (int i = 0; i < size; i++) {
        kernel[i] = (float *)malloc(size * sizeof(float));
    }
    int radius = size / 2;
    float sum = 0.0f;
    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            float value = exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            kernel[x + radius][y + radius] = value;
            sum += value;         
        }
    }    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel[i][j] /= sum;       
        }       
    }
    return kernel;
}

unsigned char* cpuGauss(unsigned char* img, int width, int height, int channels, int kernel_size, float sigma) {
    // Generate kernel
    float** kernel = kernel_gen(kernel_size, sigma);
    int radius = kernel_size / 2;
    
    // Allocate memory for output
    size_t osize = width * height * channels;
    unsigned char* output = (unsigned char*)malloc(osize);
    
    // Loop over pixels
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Initialize color channels to 0
            float c = 0.0f;
            // Loop over kernel
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    // Compute pixel coordinates
                    int px = x + kx;
                    int py = y + ky;
                    // Check if pixel is within bounds
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        // Get pixel value and corresponding kernel value
                        unsigned char* p = img + (py * width + px) * channels;
                        float kernel_value = kernel[kx + radius][ky + radius];
                        // Accumulate color channel values
                        c += *(p) * kernel_value;
                                            }
                }
            }
            // Set output pixel value
            unsigned char* pb = output + (y * width + x) * channels;
            *pb = (unsigned char)c;
            *(pb + 1) = *(img + (y * width + x) * channels + 1);
        }
    }
    
    // Free kernel memory
    for (int i = 0; i < kernel_size; i++) {
        free(kernel[i]);
    }
    free(kernel);
    
    return output;
}
unsigned char* cpuBrightness(unsigned char* img,  int width,  int height,  int channels, unsigned char delta, uint8_t darken = 0){
    size_t osize = width * height * channels;
    unsigned char* output = (unsigned char*)malloc(osize);
    for (unsigned char* p = img, *pb = output; p < img + width*height*channels; p+=channels, pb+=channels){
        unsigned int new_r = *(p) + delta;
        unsigned int new_g = *(p + 1) + delta;
        unsigned int new_b = *(p + 2) + delta;
        unsigned int new_alpha = *(p + 3);
        if (new_r < 255){
            *pb = (unsigned char) new_r;
        }
        else{
            *pb = 255;
        }
        if (new_g < 255){
            *(pb+1) = (unsigned char) new_g;
        }
        else{
            *(pb+1) = 255;
        }
        if (new_b < 255){
            *(pb+2) = (unsigned char) new_b;
        }
        else{
            *(pb+2) = 255;
        }
        *(pb+3) = new_alpha;
    }
    return output;
}

unsigned char* cpuGrayscale(unsigned char* img,  int width,  int height,  int channels){
    size_t osize = width * height * GREYCHAN;
    unsigned char* output = (unsigned char*)malloc(osize);
    for (unsigned char* p = img, *pb = output; p < img + width*height*channels; p+=channels, pb+=GREYCHAN){
        unsigned int new_val = *(p) * 0.299f 
                            + *(p + 1) * 0.587f
                            + *(p + 2) * 0.114f;
        if (new_val > 255){new_val = 255;}
        *(pb) = new_val;
        unsigned int new_alpha = *(p + 3);
        *(pb+1) = new_alpha;
    }
    return output;
}
unsigned char* cpuErode(unsigned char* img, int width, int height) {
    size_t osize = width * height * GREYCHAN;
    unsigned char* output = (unsigned char*)malloc(osize);
    const int kSize = 3; // kernel size
    const int kOffset = kSize / 2; // kernel offset
    const int threshold = 1; // threshold for erosion
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int count = 0;
            for (int ky = -kOffset; ky <= kOffset; ++ky) {
                for (int kx = -kOffset; kx <= kOffset; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        unsigned char pixel = *(img + ny * width *GREYCHAN + nx * GREYCHAN);
                        if (pixel == 255) {
                            ++count;
                        }
                    }
                }
            }
            *(output + y * width * GREYCHAN + x* GREYCHAN) = (count >= threshold) ? 255 : 0;
            *(output + y * width * GREYCHAN + x* GREYCHAN + 1) = 255;
        }
    }
    return output;
}
unsigned char* cpuDilate(unsigned char* img, int width, int height) {
    size_t osize = width * height * GREYCHAN;
    unsigned char* output = (unsigned char*)malloc(osize);
    const int kSize = 3; // kernel size
    const int kOffset = kSize / 2; // kernel offset
    const int threshold = 1; // threshold for dilate
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int count = 0;
            for (int ky = -kOffset; ky <= kOffset; ++ky) {
                for (int kx = -kOffset; kx <= kOffset; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        unsigned char pixel = *(img + ny * width *GREYCHAN + nx * GREYCHAN);
                        if (pixel == 0) {
                            ++count;
                        }
                    }
                }
            }
            *(output + y * width * GREYCHAN + x* GREYCHAN) = (count >= threshold) ? 0 : 255;
            *(output + y * width * GREYCHAN + x* GREYCHAN + 1) = 255;
        }
    }
    return output;
}

unsigned char* cpuThreshold(unsigned char* img,  int width,  int height,  int channels){
    size_t osize = width * height * GREYCHAN;
    unsigned char* output = (unsigned char*)malloc(osize);
    for (unsigned char* p = img, *pb = output; p < img + width*height*channels; p+=channels, pb+=GREYCHAN){
        unsigned int new_val = *(p) * 0.299f 
                            + *(p + 1) * 0.587f
                            + *(p + 2) * 0.114f;
       if (new_val >100){
            new_val = 255;
        }
        else{
            new_val = 0;
        }
        *(pb) = new_val;
        unsigned int new_alpha = *(p + 3);
        *(pb+1) = new_alpha;
    }
    return output;
}
unsigned char* cpuSobel(unsigned char* input, int width, int height) {
    size_t osize = width * height * GREYCHAN;
    unsigned char* output = (unsigned char*)malloc(osize);
    float kernelx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    float kernely[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Initialize color channels to 0
            float gx = 0.0f, gy = 0.0f;
            
            
            // Loop over kernel
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    // Compute pixel coordinates
                    int px = x + kx;
                    int py = y + ky;
                    // Check if pixel is within bounds
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        // Get pixel value and corresponding kernel value
                        unsigned char* p = input + (py * width + px) * GREYCHAN;
                        float kernelx_value = kernelx[kx + 1][ky + 1];
                        float kernely_value = kernely[kx + 1][ky + 1];
                        // Accumulate color channel values
                        gx += *(p) * kernelx_value;
                        gy += *(p) * kernely_value;
                    }
                }
            }
            // Compute the magnitude and direction of the gradient
            float magnitude = sqrt(gx * gx + gy * gy);
            // Set output pixel value
            unsigned char* pb = output + (y * width + x) * GREYCHAN;
            *pb = (unsigned char)magnitude;
            *(pb + 1) = *(input + (y * width + x) * GREYCHAN + 1);
        }
    }
    return output;
}


int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " brighten # OR grayscale OR gauss OR erode OR dilate or Sobel" << endl;
        return 1;
    }

    //int brightness = 30;
    char command1[9]="brighten";
    char command2[10]="grayscale";
    char command3[6]="gauss";
    char command4[6]="erode";
    char command5[7]="dilate";
    char command6[6]="sobel";
    /*Mat img = imread("img.png", IMREAD_COLOR);
    if (img.empty()) {
        cout << "Failed to load image" << endl;
        return 1;
    }

    Mat img_out_brighten(img.size(), img.type());

    int rows = img.rows;
    int cols = img.cols;
    int channels = img.channels();

    unsigned char *data = img.data;
    unsigned char *data_out_brighten = img_out_brighten.data;*/
    //Read image with STB for Uzair's CPU functions
    int width, height, channels;
    unsigned char* img = stbi_load("img.png", &width, &height, &channels, 0);
    if(img == NULL){
        printf("did not load img\n");
        exit(1);
    }
    else{
        printf("loaded img of %d width, %d height, %d channels \n", width, height, channels);
    }

    // Call the CUDA kernel function
    if(strcmp(argv[1],command1) == 0){
        long x;
        if (argc < 3){
            cout << "ERROR: include a third integer argument for brightness" << endl;
        }
        char bright = (char)strtol(argv[2], NULL, 10);
        cout <<"brightness change: " << +bright << endl;

        //run and time CPU function
        auto start = high_resolution_clock::now();
        unsigned char* output = cpuBrightness(img, width, height, channels, bright);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout<<"CPU Brightness Time in microsecs: " << duration.count() << std::endl;
        stbi_write_png("cpuoutput.png", width, height, channels, output, width*channels);

        //Call GPU function, timing happens inside function
        unsigned char* gpuoutput = brighten_image(img, width, height, channels, bright);
        stbi_write_png("gpuoutput.png", width, height, channels, gpuoutput, width*channels);
        printf("brightening\n");
    }
    else if(strcmp(argv[1],command2) == 0){
        //run and time CPU function
        auto start = high_resolution_clock::now();
        unsigned char* output = cpuGrayscale(img, width, height, channels);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout<<"CPU Grayscale Time in microsecs: " << duration.count() << std::endl;
        stbi_write_png("cpuoutput.png", width, height, GREYCHAN, output, width*GREYCHAN);

        //Call GPU function, timing happens inside function
        unsigned char* gpuoutput = grayscale_image(img, width, height, channels);
        stbi_write_png("gpuoutput.png", width, height, GREYCHAN, gpuoutput, width*GREYCHAN);
        printf("grayscaling\n");
    }
    else if(strcmp(argv[1],command3) == 0){
        //gauss_kernel(FILTERSIZE, SIGMA);
        //run and time CPU function
        unsigned char* goutput = cpuGrayscale(img, width, height, channels);
        auto start = high_resolution_clock::now();
        unsigned char* output = cpuGauss(goutput, width, height, GREYCHAN, FILTERSIZE, SIGMA);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout<<"CPU Gauss Time in microsecs: " << duration.count() << std::endl;
        stbi_write_png("cpuoutput.png", width, height, GREYCHAN, output, width*GREYCHAN);

        unsigned char* graygpu = grayscale_image(img, width, height, channels);
        unsigned char* gpuoutput = cudaGauss(graygpu, width, height, FILTERSIZE, SIGMA);
        stbi_write_png("gpuoutput.png", width, height, GREYCHAN, gpuoutput, width*GREYCHAN); 
    
        printf("gaussian blurring\n"); 
    }
    else if(strcmp(argv[1],command4) == 0){
        
        //convert to grayscale then threshold on CPU and time
        auto start = high_resolution_clock::now();
        unsigned char* binary = cpuThreshold(img, width, height, channels);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout<<"CPU Threshold Time in microsecs: " << duration.count() << std::endl;
        //write the binary image so we can see and compare
        stbi_write_png("binary.png", width, height, GREYCHAN, binary, width*GREYCHAN);


        //convert to grayscale and threshold on GPU and time
        
        unsigned char* cudabinary = cudaThreshold(img, width, height);
        //write so we can look
        stbi_write_png("cudabinary.png", width, height, GREYCHAN, cudabinary, width*GREYCHAN);
        

        //run and time CPU erode function
        auto start2 = high_resolution_clock::now();
        unsigned char* output = cpuErode(binary, width, height);
        auto stop2 = high_resolution_clock::now();
        auto duration2 = duration_cast<microseconds>(stop2 - start2);
        std::cout<<"CPU Erode Time in microsecs: " << duration2.count() << std::endl;
        stbi_write_png("cpuoutput.png", width, height, GREYCHAN, output, width*GREYCHAN);

       
        //Run and time GPU erode function - using CPU binary file right now.
        //Change binary to cudabinary to use GPU binary or to binary to use CPU binary
        //timing happens inside function
        unsigned char* gpuoutput = cudaErode(cudabinary, width, height);
        stbi_write_png("gpuoutput.png", width, height, GREYCHAN, gpuoutput, width*GREYCHAN); 
        printf("eroding\n");
    }
    else if(strcmp(argv[1],command5) == 0){
        
        //convert to grayscale then threshold on CPU and time
        auto start = high_resolution_clock::now();
        unsigned char* binary = cpuThreshold(img, width, height, channels);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout<<"CPU Threshold Time in microsecs: " << duration.count() << std::endl;
        //write the binary image so we can see and compare
        stbi_write_png("binary.png", width, height, GREYCHAN, binary, width*GREYCHAN);


        //convert to grayscale and threshold on GPU and time
        
        unsigned char* cudabinary = cudaThreshold(img, width, height);
        //write so we can look
        stbi_write_png("cudabinary.png", width, height, GREYCHAN, cudabinary, width*GREYCHAN);
        

        //run and time CPU dilate function
        auto start2 = high_resolution_clock::now();
        unsigned char* output = cpuDilate(binary, width, height);
        auto stop2 = high_resolution_clock::now();
        auto duration2 = duration_cast<microseconds>(stop2 - start2);
        std::cout<<"CPU Dilate Time in microsecs: " << duration2.count() << std::endl;
        stbi_write_png("cpuoutput.png", width, height, GREYCHAN, output, width*GREYCHAN);

       
        //Run and time GPU dilate function - using CPU binary file right now.
        //Change binary to cudabinary to use GPU binary or to binary to use CPU binary
        //timing happens inside function
        unsigned char* gpuoutput = cudaDilate(cudabinary, width, height);
        stbi_write_png("gpuoutput.png", width, height, GREYCHAN, gpuoutput, width*GREYCHAN); 
        printf("dilating\n");
    }
    else if(strcmp(argv[1],command6) == 0){
        //gauss_kernel(FILTERSIZE, SIGMA);
        //run and time CPU function
        auto start3 = high_resolution_clock::now();
        unsigned char* goutput = cpuGrayscale(img, width, height, channels);
        auto stop3 = high_resolution_clock::now();
        auto duration3 = duration_cast<microseconds>(stop3 - start3);
        std::cout<<"CPU Grayscale Time in microsecs: " << duration3.count() << std::endl;
        auto start = high_resolution_clock::now();
        unsigned char* gaussoutput = cpuGauss(goutput, width, height, GREYCHAN, FILTERSIZE, SIGMA);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout<<"CPU Gauss Time in microsecs: " << duration.count() << std::endl;

        auto start2 = high_resolution_clock::now();
        unsigned char* output = cpuSobel(gaussoutput, width, height);
        auto stop2 = high_resolution_clock::now();
        auto duration2 = duration_cast<microseconds>(stop2 - start2);
        std::cout<<"CPU Sobel Time in microsecs: " << duration2.count() << std::endl;
        
        stbi_write_png("cpuoutput.png", width, height, GREYCHAN, output, width*GREYCHAN);


        unsigned char* graygpu = grayscale_image(img, width, height, channels);
        unsigned char* blurgpu = cudaGauss(graygpu, width, height, FILTERSIZE, SIGMA);
        unsigned char* gpuoutput = cudaSobel(blurgpu, width, height);
        stbi_write_png("gpuoutput.png", width, height, GREYCHAN, gpuoutput, width*GREYCHAN); 
    
        printf("applying sobel\n"); 
    }

    return 0;
}

 //try dilate
    //    unsigned char* eoutput = cpuDilate(binary, width, height);
      //  stbi_write_png("dilatecpuoutput.png", width, height, 2, eoutput, width*2);
