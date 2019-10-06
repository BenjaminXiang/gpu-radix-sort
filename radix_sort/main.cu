#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <ctime>

#include "sort.h"
#include "utils.h"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

void cpu_sort(unsigned int* h_out, unsigned int* h_in, size_t len)
{
    for (int i = 0; i < len; ++i)
    {
        h_out[i] = h_in[i];
    }

    std::sort(h_out, h_out + len);
}

void test_cpu_vs_gpu(unsigned int* h_in, unsigned int num_elems)
{
    std::clock_t start;

    unsigned int* h_out_cpu = new unsigned int[num_elems];
    unsigned int* h_out_gpu = new unsigned int[num_elems];

    start = std::clock();
    cpu_sort(h_out_cpu, h_in, num_elems);
    double cpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    std::cout << "CPU time: " << cpu_duration << " s" << std::endl;
    
    unsigned int* d_in;
    unsigned int* d_out;
    checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned int) * num_elems));
    checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned int) * num_elems));
    checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(unsigned int) * num_elems, cudaMemcpyHostToDevice));
    start = std::clock();
    radix_sort(d_out, d_in, num_elems);
    double gpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    std::cout << "GPU time: " << gpu_duration << " s" << std::endl;
    checkCudaErrors(cudaMemcpy(h_out_gpu, d_out, sizeof(unsigned int) * num_elems, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaFree(d_in));

    cudaEvent_t event_start, stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&stop);

    unsigned int* d_thrust_in;
    checkCudaErrors( cudaMalloc(&d_thrust_in, sizeof(unsigned int) * num_elems) );
    checkCudaErrors( cudaMemcpy(d_thrust_in, h_in, sizeof(unsigned int) * num_elems, cudaMemcpyHostToDevice) );

    thrust::device_ptr<unsigned int> d_ptr_in(d_thrust_in);

    cudaEventRecord(event_start);
    thrust::sort(d_ptr_in, d_ptr_in + num_elems);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, event_start, stop);
    milliseconds /= 1000.0;
    std::cout << "Thrust time is " << milliseconds << std::endl;
    std::cout << "Thrust Speedup cpu " << cpu_duration / milliseconds << "x" << std::endl; 
    std::cout << "Thrust Speedup 4-way " << gpu_duration / milliseconds << "x" << std::endl;

    // Calculate GPU / CPU speedup
    std::cout << "Speedup: " << cpu_duration / gpu_duration << "x" << std::endl;

    // Check for any mismatches between outputs of CPU and GPU
    bool match = true;
    int index_diff = 0;
    for (int i = 0; i < num_elems; ++i)
    {
        if (h_out_cpu[i] != h_out_gpu[i])
        {
            match = false;
            index_diff = i;
            break;
        }
    }
    std::cout << "Match: " << match << std::endl;
    
    // Detail the mismatch if any
    if (!match)
    {
        std::cout << "Difference in index: " << index_diff << std::endl;
        std::cout << "CPU: " << h_out_cpu[index_diff] << std::endl;
        std::cout << "GPU Radix Sort: " << h_out_gpu[index_diff] << std::endl;
        int window_sz = 10;
    
        std::cout << "Contents: " << std::endl;
        std::cout << "CPU: ";
        for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
        {
            std::cout << h_out_cpu[index_diff + i] << ", ";
        }
        std::cout << std::endl;
        std::cout << "GPU Radix Sort: ";
        for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
        {
            std::cout << h_out_gpu[index_diff + i] << ", ";
        }
        std::cout << std::endl;
    }
    
    delete[] h_out_gpu;
    delete[] h_out_cpu;
}

int main()
{
    // Set up clock for timing comparisons
    srand(1);

    for (int i = 20; i < 21; ++i)
    {
        unsigned int num_elems = (1 << i);
        //unsigned int num_elems = 8;
        std::cout << "h_in size: " << num_elems << std::endl;

        unsigned int* h_in = new unsigned int[num_elems];
        unsigned int* h_in_rand = new unsigned int[num_elems];

        for (int j = 0; j < num_elems; j++)
        {
            h_in[j] = (num_elems - 1) - j;
            h_in_rand[j] = rand() % num_elems;
            //std::cout << h_in[j] << " ";
        }
        //std::cout << std::endl;

        std::cout << "*** i: " << i << " ***" << std::endl;
        for (int j = 0; j < 5; ++j) {
            std::cout << "*****Descending order*****" << std::endl;
            test_cpu_vs_gpu(h_in, num_elems);
            std::cout << "*****Random order*****" << std::endl;
            test_cpu_vs_gpu(h_in_rand, num_elems);
            std::cout << std::endl;
        }

        delete[] h_in;
        delete[] h_in_rand;

        std::cout << std::endl;
    }
}
