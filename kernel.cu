#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "kernel.h"
#include <cmath>

using namespace std;
using namespace cv;
#include <chrono>

#define BLOCK_SIZE 32
#define CHANNELS 4
#define TILE_SIZE 32
#define GREYCHAN 2
#define FILTERSIZE 15
#define SIGMA 4.0f

__global__ void brighten_kernel(const unsigned char *input, unsigned char *output, int height, int width, int channels, int brightness)
{
    __shared__ unsigned char tile[TILE_SIZE][TILE_SIZE][CHANNELS];
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < width && y < height)
    {
        // Load tile of image input into shared memory
        int idx = (y * width + x) * channels;
        for (int c = 0; c < channels; c++)
        {
            tile[threadIdx.y][threadIdx.x][c] = input[idx + c];
        }

        // Wait for all threads to finish loading
        __syncthreads();

        // Brighten the pixel
        for (int c = 0; c < channels; c++)
        {
            int val = tile[threadIdx.y][threadIdx.x][c] + brightness;
            val = min(val, 255);
            output[idx + c] = val;
        }
    }
}

unsigned char* brighten_image(unsigned char* img, int width, int height, int channels, int brightness) {
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    size_t osize = width * height * channels;
    unsigned char* output = (unsigned char*)malloc(osize);

    unsigned char *dev_data;
    unsigned char *dev_data_out;

    cudaMalloc(&dev_data, height * width * channels);
    cudaMalloc(&dev_data_out, height * width * channels);
    
    
    cudaMemcpy(dev_data, img, height * width * channels, cudaMemcpyHostToDevice);
    auto start = std::chrono::high_resolution_clock::now();
    brighten_kernel<<<grid_size, block_size>>>(dev_data, dev_data_out, height, width, channels, brightness);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cudaMemcpy(output, dev_data_out, height * width * channels, cudaMemcpyDeviceToHost);
    
    std::cout<<"GPU Brightness Time in microsecs: " <<duration.count() << std::endl;
    cudaFree(dev_data);
    cudaFree(dev_data_out);
    return output;
}

__global__ void grayscale_kernel(unsigned char *input, unsigned char *output, int height, int width, int channels)
{
	__shared__ unsigned char tile[TILE_SIZE][TILE_SIZE][CHANNELS];
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < width && y < height)
    {
        // Load tile of image input into shared memory
        int idx = (y * width + x) * channels;
        int outidx = (y*width +x) * GREYCHAN;
        for (int c = 0; c < channels; c++)
        {
            tile[threadIdx.y][threadIdx.x][c] = input[idx + c];
        }

        // Wait for all threads to finish loading
        __syncthreads();

        // Brighten the pixel
        //I unrolled this loop
        //for (int c = 0; c < channels; c++)
        //{
            int val = tile[threadIdx.y][threadIdx.x][0] * 0.299f + tile[threadIdx.y][threadIdx.x][1] * 0.587f + tile[threadIdx.y][threadIdx.x][2] * 0.114f;
            int alpha = tile[threadIdx.y][threadIdx.x][3];
            val = min(val, 255);
            output[outidx] = val;
            output[outidx + 1] = alpha;
        //}
    }
}

unsigned char* grayscale_image(unsigned char* img, int width, int height, int channels) {
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    unsigned char *dev_data;
    unsigned char *dev_data_out;

    size_t osize = width * height * GREYCHAN;
    unsigned char* output = (unsigned char*)malloc(osize);

    cudaMalloc(&dev_data, height * width * channels);
    cudaMalloc(&dev_data_out, height * width * GREYCHAN);
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dev_data, img, height * width * channels, cudaMemcpyHostToDevice);
    
    grayscale_kernel<<<grid_size, block_size>>>(dev_data, dev_data_out, height, width, channels);
    
    cudaMemcpy(output, dev_data_out, height * width * GREYCHAN, cudaMemcpyDeviceToHost);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout<<"GPU Grayscale Time in microsecs: " << duration.count() << std::endl;
    cudaFree(dev_data);
    cudaFree(dev_data_out);
    return output;
}

//REPLACE ERODE KERNEL
__global__ void erode_kernel(const unsigned char *input, unsigned char *output, int height, int width) {
    const int kSize = 3;
    const int kOffset = kSize / 2;
    const int threshold = 1;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int count = 0;
        for (int ky = -kOffset; ky <= kOffset; ++ky) {
            for (int kx = -kOffset; kx <= kOffset; ++kx) {
                int nx = x + kx;
                int ny = y + ky;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    unsigned char pixel = *(input + ny * width * GREYCHAN + nx * GREYCHAN);
                    if (pixel==255) {
                        ++count;
                    }
                }
            }
        }
        *(output + y * width * GREYCHAN + x * GREYCHAN) = (count >= threshold) ? 255 : 1;
        *(output + y * width * GREYCHAN + x * GREYCHAN + 1) = 255;
    }
}


unsigned char* cudaErode(unsigned char* img, int width, int height) {
    
    size_t osize = width * height * GREYCHAN;
    unsigned char* output = (unsigned char*)malloc(osize);
    unsigned char *dev_input;
    unsigned char *dev_output;

    cudaMalloc(&dev_input, width * height * GREYCHAN);
    cudaMalloc(&dev_output, osize);
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dev_input, img, width * height * GREYCHAN, cudaMemcpyHostToDevice);

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    erode_kernel<<<grid_size, block_size>>>(dev_input, dev_output, height, width);
    
    cudaMemcpy(output, dev_output, osize, cudaMemcpyDeviceToHost);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout<<"GPU Erode Time in microsecs: " <<duration.count() << std::endl;
    cudaFree(dev_input);
    cudaFree(dev_output);

    return output;
}
__global__ void dilate_kernel(const unsigned char *input, unsigned char *output, int height, int width) {
    const int kSize = 3;
    const int kOffset = kSize / 2;
    const int threshold = 1;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int count = 0;
        for (int ky = -kOffset; ky <= kOffset; ++ky) {
            for (int kx = -kOffset; kx <= kOffset; ++kx) {
                int nx = x + kx;
                int ny = y + ky;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    unsigned char pixel = *(input + ny * width * GREYCHAN + nx * GREYCHAN);
                    if (pixel == 0) {
                        ++count;
                    }
                }
            }
        }
        *(output + y * width * GREYCHAN + x * GREYCHAN) = (count >= threshold) ? 0 : 255;
        *(output + y * width * GREYCHAN + x * GREYCHAN + 1) = 255;
    }
}

unsigned char* cudaDilate(unsigned char* img, int width, int height) {
    
    size_t osize = width * height * GREYCHAN;
    unsigned char* output = (unsigned char*)malloc(osize);

    unsigned char *dev_input;
    unsigned char *dev_output;

    cudaMalloc(&dev_input, width * height * GREYCHAN);
    cudaMalloc(&dev_output, osize);
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dev_input, img, width * height * GREYCHAN, cudaMemcpyHostToDevice);

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    dilate_kernel<<<grid_size, block_size>>>(dev_input, dev_output, height, width);
    
    cudaMemcpy(output, dev_output, osize, cudaMemcpyDeviceToHost);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout<<"GPU Dilate Time in microsecs: " <<duration.count() << std::endl;
    cudaFree(dev_input);
    cudaFree(dev_output);

    return output;
}
__global__ void threshold_kernel(const unsigned char* input, unsigned char* output, int width, int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned int index = y * width * 4 + x * 4;
        unsigned int new_val = input[index] * 0.299f
                            + input[index + 1] * 0.587f
                            + input[index + 2] * 0.114f;
        if (new_val > 100) {
            new_val = 255;
        }
        else {
            new_val = 0;
        }
        output[y * width * GREYCHAN + x * GREYCHAN] = new_val;
        output[y * width * GREYCHAN + x * GREYCHAN + 1] = input[index + 3];
    }
}

unsigned char* cudaThreshold(unsigned char* img, int width, int height) {
    
    size_t isize = width * height * 4 * sizeof(unsigned char);
    size_t osize = width * height * GREYCHAN * sizeof(unsigned char);

    unsigned char* output = (unsigned char*)malloc(osize);
    unsigned char *dev_input, *dev_output;

    cudaMalloc(&dev_input, isize);
    cudaMalloc(&dev_output, osize);
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dev_input, img, isize, cudaMemcpyHostToDevice);

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    threshold_kernel<<<grid_size, block_size>>>(dev_input, dev_output, width, height);
    
    cudaMemcpy(output, dev_output, osize, cudaMemcpyDeviceToHost);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout<<"GPU Theshold Time in microsecs: " <<duration.count() << std::endl;

    cudaFree(dev_input);
    cudaFree(dev_output);

    return output;
}
__global__ void gpu_kernel_gen(float *kernel, int size, float sigma) {
    int radius = size / 2;
    float sum = 0.0f;
    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            float value = exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            kernel[(x + radius) * size + (y + radius)] = value;
            sum += value;         
        }
    }    
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;       
    }   
}
__global__ void gauss_kernel(unsigned char* input, unsigned char* output, int width, int height, float* kernelog) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;
    
    int radius = FILTERSIZE / 2; 
    __shared__ float kernel[FILTERSIZE*FILTERSIZE];
    if (threadIdx.x < FILTERSIZE && threadIdx.y < FILTERSIZE){
        kernel[threadIdx.y*FILTERSIZE + threadIdx.x] = kernelog[threadIdx.y*FILTERSIZE + threadIdx.x];
    }
    __syncthreads();
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
                unsigned char* p = input + (py * width + px) * GREYCHAN;
                float kernel_value = kernel[(kx + radius) * FILTERSIZE + (ky + radius)] ;
                // Accumulate color channel values
                c += *(p) * kernel_value;
                
            }
        }
    }
    // Set output pixel value
    unsigned char* pb = output + (y * width + x) * GREYCHAN;
    *pb = (unsigned char)c;
    *(pb + 1) = *(input + (y * width + x) * GREYCHAN + 1);
}
unsigned char* cudaGauss(unsigned char* img, int width, int height, int size, float sigma) {
    size_t osize = width * height * GREYCHAN * sizeof(unsigned char);
    float* dev_kernel;
    unsigned char* output = (unsigned char*)malloc(osize);
    unsigned char *dev_input, *dev_output;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaMalloc(&dev_input, osize);
    cudaMalloc(&dev_output, osize);
    cudaMalloc(&dev_kernel, size * size * sizeof(float));

    //change this to isize if we want to accomodate multiple sizes
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dev_input, img, osize, cudaMemcpyHostToDevice);
    gpu_kernel_gen<<<1, 1>>>(dev_kernel, size, sigma);
    gauss_kernel<<<grid_size, block_size>>>(dev_input, dev_output, width, height, dev_kernel);
    auto stop = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(output, dev_output, osize, cudaMemcpyDeviceToHost);
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout<<"GPU Gauss Time in microsecs: " <<duration.count() << std::endl;

    cudaFree(dev_input);
    cudaFree(dev_output);

    cudaFree(dev_kernel);

    return output;
}
__global__ void sobel_kernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;

    // Initialize color channels to 0
    float gx = 0.0f, gy = 0.0f;
    // Sobel kernels for x and y direction, shared
    __shared__ float kernelx[3][3];
    __shared__ float kernely[3][3];
    if (threadIdx.x == 0 && threadIdx.y == 0){
        kernelx[0][0] = -1; kernelx[0][1] = 0; kernelx[0][2] = 1;
        kernelx[1][0] = -2; kernelx[1][1] = 0; kernelx[1][2] = 2;
        kernelx[2][0] = -1; kernelx[2][1] = 0; kernelx[2][2] = 1;
        
        kernely[0][0] = -1; kernely[0][1] = -2; kernely[0][2] = -1;
        kernely[1][0] = 0; kernely[1][1] = 0; kernely[1][2] = 0;
        kernely[2][0] = 1; kernely[2][1] = 2; kernely[2][2] = 1;
    }
    __syncthreads(); 
    /*nonshared im()plementation
    float kernelx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    float kernely[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};*/
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
    // Compute the magnitude of the gradient
    float magnitude = sqrt(gx * gx + gy * gy);
    // Set output pixel value
    unsigned char* pb = output + (y * width + x) * GREYCHAN;
    *pb = (unsigned char)magnitude;
    *(pb + 1) = *(input + (y * width + x) * GREYCHAN + 1);
}

unsigned char* cudaSobel(unsigned char* img, int width, int height) {
    size_t osize = width * height * GREYCHAN * sizeof(unsigned char);
    unsigned char* output = (unsigned char*)malloc(osize);
    unsigned char *dev_input, *dev_output;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaMalloc(&dev_input, osize);
    cudaMalloc(&dev_output, osize);
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dev_input, img, osize, cudaMemcpyHostToDevice);
    sobel_kernel<<<grid_size, block_size>>>(dev_input, dev_output, width, height);
    
    cudaMemcpy(output, dev_output, osize, cudaMemcpyDeviceToHost);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout<<"GPU Sobel Time in microsecs: " <<duration.count() << std::endl;
    cudaFree(dev_input);
    cudaFree(dev_output);

    return output;
}