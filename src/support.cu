#include "support.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

// ============ CPU Timer Implementation ============

void CpuTimer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

void CpuTimer::stop() {
    end_time = std::chrono::high_resolution_clock::now();
}

float CpuTimer::elapsed_ms() const {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0f;
}

// ============ GPU Timer Class ============

class GpuTimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    GpuTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    
    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        cudaEventRecord(start_event, 0);
    }
    
    void stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
    }
    
    float elapsed_ms() {
        float ms;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }
};

// ============ Device Query ============

DeviceInfo query_device_info() {
    DeviceInfo info;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    info.name = prop.name;
    info.memory_gb = prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f);
    info.sm_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.warp_size = prop.warpSize;
    
    // Peak GFLOPS (FP32)
    // Cores per SM varies by architecture, assuming 128 for modern GPUs
    int cores_per_sm = 128;
    float clock_ghz = prop.clockRate / 1e6;
    info.peak_gflops = info.sm_count * cores_per_sm * clock_ghz * 2; // *2 for FMA
    
    // Peak memory bandwidth
    float memory_clock_ghz = prop.memoryClockRate / 1e6;
    int memory_bus_width = prop.memoryBusWidth;
    info.peak_bandwidth_gbs = 2.0 * memory_clock_ghz * (memory_bus_width / 8.0) / 1000.0;
    
    return info;
}

void print_device_info(const DeviceInfo& info) {
    std::cout << "\n========== GPU Device Info ==========\n";
    std::cout << "Device:              " << info.name << "\n";
    std::cout << "Global Memory:       " << std::fixed << std::setprecision(2) 
              << info.memory_gb << " GB\n";
    std::cout << "SM Count:            " << info.sm_count << "\n";
    std::cout << "Peak GFLOPS (FP32):  " << std::setprecision(1) 
              << info.peak_gflops << "\n";
    std::cout << "Peak Bandwidth:      " << info.peak_bandwidth_gbs << " GB/s\n";
    std::cout << "Max Threads/Block:   " << info.max_threads_per_block << "\n";
    std::cout << "Warp Size:           " << info.warp_size << "\n";
    std::cout << "=====================================\n\n";
}

// ============ Metrics Calculation ============

long long calculate_bokeh_flops(int width, int height, int blur_radius,
                                 const unsigned char* mask, int ops_per_sample) {
    long long total_flops = 0;
    int kernel_size = (2 * blur_radius + 1) * (2 * blur_radius + 1);
    
    for (int i = 0; i < width * height; i++) {
        if (mask[i] == 0) {
            // Background pixel: needs blur computation
            total_flops += kernel_size * ops_per_sample;
        } else {
            // Foreground pixel: just copy (minimal ops)
            total_flops += 3; // 3 channels copy
        }
    }
    
    return total_flops;
}

long long calculate_memory_bytes(int width, int height, int channels,
                                  int blur_radius, const unsigned char* mask) {
    long long bytes = 0;
    int kernel_size = (2 * blur_radius + 1) * (2 * blur_radius + 1);
    
    for (int i = 0; i < width * height; i++) {
        if (mask[i] == 0) {
            // Background: read kernel_size pixels
            bytes += kernel_size * channels;
        } else {
            // Foreground: read 1 pixel
            bytes += channels;
        }
    }
    
    // Add output writes
    bytes += width * height * channels;
    
    return bytes;
}

// ============ Print Functions ============

void print_frame_metrics(const FrameMetrics& m, const DeviceInfo& dev) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Kernel: " << m.kernel_time_ms << " ms";
    std::cout << "  |  GFLOPS: " << std::setprecision(1) << m.gflops;
    std::cout << " (" << (m.gflops / dev.peak_gflops * 100) << "% peak)";
    std::cout << "  |  BW: " << std::setprecision(2) << m.bandwidth_gbs << " GB/s\n";
}

void print_summary(const std::vector<FrameMetrics>& all_metrics, const DeviceInfo& dev) {
    if (all_metrics.empty()) return;
    
    // Calculate averages
    float avg_kernel = 0, avg_total = 0, avg_cpu = 0;
    float avg_gflops = 0, avg_bandwidth = 0;
    
    for (const auto& m : all_metrics) {
        avg_kernel += m.kernel_time_ms;
        avg_total += m.total_gpu_time_ms;
        avg_cpu += m.cpu_time_ms;
        avg_gflops += m.gflops;
        avg_bandwidth += m.bandwidth_gbs;
    }
    
    int n = all_metrics.size();
    avg_kernel /= n;
    avg_total /= n;
    avg_cpu /= n;
    avg_gflops /= n;
    avg_bandwidth /= n;
    
    float speedup = avg_cpu / avg_kernel;
    float fps_gpu = 1000.0f / avg_total;
    float fps_cpu = 1000.0f / avg_cpu;
    
    std::cout << "\n========== Performance Summary ==========\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Frames processed:    " << n << "\n";
    std::cout << "Resolution:          " << all_metrics[0].width << "x" 
              << all_metrics[0].height << "\n";
    std::cout << "Blur radius:         " << all_metrics[0].blur_radius << "\n";
    
    std::cout << "\n--- Timing ---\n";
    std::cout << "Avg kernel time:     " << avg_kernel << " ms\n";
    std::cout << "Avg total GPU time:  " << avg_total << " ms\n";
    std::cout << "Avg CPU time:        " << avg_cpu << " ms\n";
    std::cout << "Speedup:             " << std::setprecision(1) << speedup << "x\n";
    
    std::cout << "\n--- Throughput ---\n";
    std::cout << "GPU FPS:             " << fps_gpu << "\n";
    std::cout << "CPU FPS:             " << fps_cpu << "\n";
    
    std::cout << "\n--- Compute ---\n";
    std::cout << "Avg GFLOPS:          " << avg_gflops << "\n";
    std::cout << "Peak Utilization:    " << (avg_gflops / dev.peak_gflops * 100) << "%\n";
    
    std::cout << "\n--- Memory ---\n";
    std::cout << "Avg Bandwidth:       " << avg_bandwidth << " GB/s\n";
    std::cout << "Peak Utilization:    " << (avg_bandwidth / dev.peak_bandwidth_gbs * 100) << "%\n";
    std::cout << "=========================================\n";
}

// ============ CSV Export ============

void export_csv(const std::vector<FrameMetrics>& all_metrics, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return;
    }
    
    // Header
    file << "frame,width,height,blur_radius,";
    file << "kernel_ms,memory_htod_ms,memory_dtoh_ms,total_gpu_ms,cpu_ms,";
    file << "flops,gflops,bytes_read,bytes_written,bandwidth_gbs,";
    file << "speedup\n";
    
    // Data
    for (size_t i = 0; i < all_metrics.size(); i++) {
        const auto& m = all_metrics[i];
        file << i << ",";
        file << m.width << "," << m.height << "," << m.blur_radius << ",";
        file << m.kernel_time_ms << "," << m.memory_htod_ms << ",";
        file << m.memory_dtoh_ms << "," << m.total_gpu_time_ms << ",";
        file << m.cpu_time_ms << ",";
        file << m.flop_count << "," << m.gflops << ",";
        file << m.bytes_read << "," << m.bytes_written << ",";
        file << m.bandwidth_gbs << ",";
        file << (m.cpu_time_ms / m.kernel_time_ms) << "\n";
    }
    
    file.close();
    std::cout << "Metrics exported to: " << filename << "\n";
}