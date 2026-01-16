#ifndef SUPPORT_H
#define SUPPORT_H

#include <string>
#include <vector>
#include <chrono>

// ============ Constants ============
#define DEFAULT_BLUR_RADIUS 15
#define DEFAULT_BLOCK_SIZE 16
#define MAX_BLUR_RADIUS 50

// ============ Structs ============

struct DeviceInfo {
    std::string name;
    float memory_gb;
    int sm_count;
    float peak_gflops;
    float peak_bandwidth_gbs;
    int max_threads_per_block;
    int warp_size;
};

struct FrameMetrics {
    // Timing (milliseconds)
    float kernel_time_ms;
    float memory_htod_ms;
    float memory_dtoh_ms;
    float total_gpu_time_ms;
    float cpu_time_ms;
    
    // Compute
    long long flop_count;
    float gflops;
    
    // Memory
    long long bytes_read;
    long long bytes_written;
    float bandwidth_gbs;
    
    // Frame info
    int width;
    int height;
    int blur_radius;
};

// ============ Device Functions ============
DeviceInfo query_device_info();
void print_device_info(const DeviceInfo& info);

// ============ Metrics Functions ============
long long calculate_bokeh_flops(int width, int height, int blur_radius, 
                                 const unsigned char* mask, int ops_per_sample);

long long calculate_memory_bytes(int width, int height, int channels,
                                  int blur_radius, const unsigned char* mask);

// ============ Output Functions ============
void print_frame_metrics(const FrameMetrics& m, const DeviceInfo& dev);
void print_summary(const std::vector<FrameMetrics>& all_metrics, const DeviceInfo& dev);
void export_csv(const std::vector<FrameMetrics>& all_metrics, const std::string& filename);

// ============ Timer Class ============
class CpuTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
public:
    void start();
    void stop();
    float elapsed_ms() const;
};

#endif