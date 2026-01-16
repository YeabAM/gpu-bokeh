#include "support.h"
#include <cuda_runtime.h>

// ============ Naive GPU Kernel ============

__global__ void bokeh_kernel_naive(unsigned char* input, 
                                    unsigned char* mask,
                                    unsigned char* output,
                                    int width, 
                                    int height,
                                    int blur_radius) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Foreground pixel: copy as-is
    if (mask[idx] == 255) {
        output[idx * 3 + 0] = input[idx * 3 + 0];
        output[idx * 3 + 1] = input[idx * 3 + 1];
        output[idx * 3 + 2] = input[idx * 3 + 2];
        return;
    }
    
    // Background pixel: apply box blur
    int sum_r = 0, sum_g = 0, sum_b = 0;
    int count = 0;
    
    for (int dy = -blur_radius; dy <= blur_radius; dy++) {
        for (int dx = -blur_radius; dx <= blur_radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            // Boundary check
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int neighbor_idx = ny * width + nx;
                sum_r += input[neighbor_idx * 3 + 0];
                sum_g += input[neighbor_idx * 3 + 1];
                sum_b += input[neighbor_idx * 3 + 2];
                count++;
            }
        }
    }
    
    output[idx * 3 + 0] = sum_r / count;
    output[idx * 3 + 1] = sum_g / count;
    output[idx * 3 + 2] = sum_b / count;
}

// ============ Shared Memory Kernel (placeholder) ============

__global__ void bokeh_kernel_shared(unsigned char* input,
                                     unsigned char* mask,
                                     unsigned char* output,
                                     int width,
                                     int height,
                                     int blur_radius) {
    // TODO: Implement later
}