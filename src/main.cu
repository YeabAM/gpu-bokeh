#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <dirent.h>

#include <cuda_runtime.h>

#include "support.h"

// stb image libraries (implementation in this file)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ============ CUDA Error Checking ============

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "Kernel Launch Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            exit(EXIT_FAILURE); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            std::cerr << "Kernel Execution Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============ Kernel Declarations ============

extern __global__ void bokeh_kernel_naive(unsigned char* input,
                                           unsigned char* mask,
                                           unsigned char* output,
                                           int width,
                                           int height,
                                           int blur_radius);

extern __global__ void bokeh_kernel_shared(unsigned char* input,
                                            unsigned char* mask,
                                            unsigned char* output,
                                            int width,
                                            int height,
                                            int blur_radius);

// ============ Helper: Count frames in directory ============

int count_frames(const std::string& dir_path) {
    int count = 0;
    DIR* dir = opendir(dir_path.c_str());
    if (dir == nullptr) {
        std::cerr << "Error: Could not open directory " << dir_path << "\n";
        return -1;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.find(".jpg") != std::string::npos || 
            name.find(".png") != std::string::npos) {
            count++;
        }
    }
    closedir(dir);
    return count;
}

// ============ Helper: Build frame filename ============

std::string frame_filename(int index, const std::string& ext) {
    std::ostringstream ss;
    ss << std::setw(5) << std::setfill('0') << index << ext;
    return ss.str();
}

// ============ Helper: Check if directory exists ============

bool directory_exists(const std::string& path) {
    DIR* dir = opendir(path.c_str());
    if (dir) {
        closedir(dir);
        return true;
    }
    return false;
}

// ============ Main ============

int main(int argc, char** argv) {
    
    // Default parameters
    std::string input_dir = "data";
    std::string output_dir = "output";
    int blur_radius = DEFAULT_BLUR_RADIUS;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_dir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--blur-radius" && i + 1 < argc) {
            blur_radius = std::stoi(argv[++i]);
            if (blur_radius <= 0 || blur_radius > MAX_BLUR_RADIUS) {
                std::cerr << "Error: Blur radius must be between 1 and " 
                          << MAX_BLUR_RADIUS << "\n";
                return 1;
            }
        } else if (arg == "--help") {
            std::cout << "Usage: ./bokeh [options]\n";
            std::cout << "  --input <dir>       Input data directory (default: data)\n";
            std::cout << "  --output <dir>      Output directory (default: output)\n";
            std::cout << "  --blur-radius <n>   Blur radius in pixels (default: 15)\n";
            return 0;
        }
    }
    
    // Validate directories
    std::string jpeg_dir = input_dir + "/JPEGImages";
    std::string mask_dir = input_dir + "/Annotations";
    
    if (!directory_exists(jpeg_dir)) {
        std::cerr << "Error: JPEGImages directory not found: " << jpeg_dir << "\n";
        return 1;
    }
    
    if (!directory_exists(mask_dir)) {
        std::cerr << "Error: Annotations directory not found: " << mask_dir << "\n";
        return 1;
    }
    
    if (!directory_exists(output_dir)) {
        std::cerr << "Error: Output directory not found: " << output_dir << "\n";
        std::cerr << "Please create it: mkdir -p " << output_dir << "\n";
        return 1;
    }
    
    // Count frames
    int num_frames = count_frames(jpeg_dir);
    if (num_frames <= 0) {
        std::cerr << "Error: No frames found in " << jpeg_dir << "\n";
        return 1;
    }
    std::cout << "Found " << num_frames << " frames\n";
    
    // Check CUDA device
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "Error: No CUDA-capable device found\n";
        return 1;
    }
    
    // Query and print device info
    DeviceInfo dev_info = query_device_info();
    print_device_info(dev_info);
    
    // Load first frame to get dimensions
    std::string first_frame_path = jpeg_dir + "/" + frame_filename(0, ".jpg");
    int width, height, channels;
    unsigned char* test_img = stbi_load(first_frame_path.c_str(), &width, &height, &channels, 3);
    
    if (!test_img) {
        std::cerr << "Error: Could not load " << first_frame_path << "\n";
        return 1;
    }
    stbi_image_free(test_img);
    
    // Force 3 channels (RGB)
    channels = 3;
    
    std::cout << "Resolution: " << width << "x" << height << "\n";
    std::cout << "Blur radius: " << blur_radius << "\n\n";
    
    // Allocate host memory
    int img_size = width * height * channels;
    int mask_size = width * height;
    
    unsigned char* h_input = new (std::nothrow) unsigned char[img_size];
    unsigned char* h_mask = new (std::nothrow) unsigned char[mask_size];
    unsigned char* h_output = new (std::nothrow) unsigned char[img_size];
    
    if (!h_input || !h_mask || !h_output) {
        std::cerr << "Error: Failed to allocate host memory\n";
        delete[] h_input;
        delete[] h_mask;
        delete[] h_output;
        return 1;
    }
    
    // Allocate device memory
    unsigned char *d_input = nullptr;
    unsigned char *d_mask = nullptr;
    unsigned char *d_output = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_input, img_size));
    CUDA_CHECK(cudaMalloc(&d_mask, mask_size));
    CUDA_CHECK(cudaMalloc(&d_output, img_size));
    
    // CUDA events for timing
    cudaEvent_t start_htod, stop_htod;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_dtoh, stop_dtoh;
    
    CUDA_CHECK(cudaEventCreate(&start_htod));
    CUDA_CHECK(cudaEventCreate(&stop_htod));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    CUDA_CHECK(cudaEventCreate(&start_dtoh));
    CUDA_CHECK(cudaEventCreate(&stop_dtoh));
    
    // Kernel configuration
    dim3 block(DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    // Metrics storage
    std::vector<FrameMetrics> all_metrics;
    int skipped_frames = 0;
    
    // Process loop
    std::cout << "Processing frames...\n";
    
    for (int i = 0; i < num_frames; i++) {
        
        // Build paths
        std::string frame_path = jpeg_dir + "/" + frame_filename(i, ".jpg");
        std::string mask_path = mask_dir + "/" + frame_filename(i, ".png");
        std::string output_path = output_dir + "/" + frame_filename(i, ".jpg");
        
        // Load frame (force RGB)
        int fw, fh, fc;
        unsigned char* frame_data = stbi_load(frame_path.c_str(), &fw, &fh, &fc, 3);
        
        if (!frame_data) {
            std::cerr << "Warning: Could not load frame " << frame_path << "\n";
            skipped_frames++;
            continue;
        }
        
        // Load mask (grayscale)
        int mw, mh, mc;
        unsigned char* mask_data = stbi_load(mask_path.c_str(), &mw, &mh, &mc, 1);
        
        if (!mask_data) {
            std::cerr << "Warning: Could not load mask " << mask_path << "\n";
            stbi_image_free(frame_data);
            skipped_frames++;
            continue;
        }
        
        // Validate dimensions
        if (fw != width || fh != height) {
            std::cerr << "Warning: Frame " << i << " has different dimensions, skipping\n";
            stbi_image_free(frame_data);
            stbi_image_free(mask_data);
            skipped_frames++;
            continue;
        }
        
        if (mw != width || mh != height) {
            std::cerr << "Warning: Mask " << i << " has different dimensions, skipping\n";
            stbi_image_free(frame_data);
            stbi_image_free(mask_data);
            skipped_frames++;
            continue;
        }
        
        // Copy to host arrays
        memcpy(h_input, frame_data, img_size);
        memcpy(h_mask, mask_data, mask_size);
        
        // Free stb loaded data
        stbi_image_free(frame_data);
        stbi_image_free(mask_data);
        
        // Host to Device
        CUDA_CHECK(cudaEventRecord(start_htod));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mask, h_mask, mask_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop_htod));
        
        // Launch kernel
        CUDA_CHECK(cudaEventRecord(start_kernel));
        bokeh_kernel_naive<<<grid, block>>>(d_input, d_mask, d_output,
                                             width, height, blur_radius);
        CUDA_KERNEL_CHECK();
        CUDA_CHECK(cudaEventRecord(stop_kernel));
        
        // Device to Host
        CUDA_CHECK(cudaEventRecord(start_dtoh));
        CUDA_CHECK(cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop_dtoh));
        
        CUDA_CHECK(cudaEventSynchronize(stop_dtoh));
        
        // Get timing
        float time_htod, time_kernel, time_dtoh;
        CUDA_CHECK(cudaEventElapsedTime(&time_htod, start_htod, stop_htod));
        CUDA_CHECK(cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel));
        CUDA_CHECK(cudaEventElapsedTime(&time_dtoh, start_dtoh, stop_dtoh));
        
        // Store metrics
        FrameMetrics metrics;
        metrics.kernel_time_ms = time_kernel;
        metrics.memory_htod_ms = time_htod;
        metrics.memory_dtoh_ms = time_dtoh;
        metrics.total_gpu_time_ms = time_htod + time_kernel + time_dtoh;
        metrics.width = width;
        metrics.height = height;
        metrics.blur_radius = blur_radius;
        
        // Calculate FLOPS and bandwidth
        metrics.flop_count = calculate_bokeh_flops(width, height, blur_radius,
                                                    h_mask, 8);
        metrics.gflops = (metrics.flop_count / 1e9) / (time_kernel / 1000.0);
        
        metrics.bytes_read = calculate_memory_bytes(width, height, channels,
                                                     blur_radius, h_mask);
        metrics.bytes_written = img_size;
        metrics.bandwidth_gbs = ((metrics.bytes_read + metrics.bytes_written) / 1e9) 
                                 / (time_kernel / 1000.0);
        
        all_metrics.push_back(metrics);
        
        // Save output frame
        int write_success = stbi_write_jpg(output_path.c_str(), width, height, 
                                            channels, h_output, 95);
        if (!write_success) {
            std::cerr << "Warning: Could not save " << output_path << "\n";
        }
        
        // Progress
        if ((i + 1) % 10 == 0 || i == num_frames - 1) {
            std::cout << "  Processed " << (i + 1) << "/" << num_frames << " frames\n";
        }
    }
    
    // Check if any frames were processed
    if (all_metrics.empty()) {
        std::cerr << "Error: No frames were successfully processed\n";
        
        delete[] h_input;
        delete[] h_mask;
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_mask);
        cudaFree(d_output);
        
        return 1;
    }
    
    // Print summary
    if (skipped_frames > 0) {
        std::cout << "\nWarning: Skipped " << skipped_frames << " frames\n";
    }
    
    print_summary(all_metrics, dev_info);
    
    // Export CSV
    export_csv(all_metrics, output_dir + "/performance_report.csv");
    
    // Cleanup
    delete[] h_input;
    delete[] h_mask;
    delete[] h_output;
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_output));
    
    CUDA_CHECK(cudaEventDestroy(start_htod));
    CUDA_CHECK(cudaEventDestroy(stop_htod));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));
    CUDA_CHECK(cudaEventDestroy(start_dtoh));
    CUDA_CHECK(cudaEventDestroy(stop_dtoh));
    
    std::cout << "\nDone! Frames saved to " << output_dir << "/\n";
    
    return 0;
}