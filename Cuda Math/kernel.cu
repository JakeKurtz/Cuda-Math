
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "Vector.h"
#include "Random.h"
#include "Math.h"
#include "Matrix.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    printf("%f\n", jek::rand_float(1, 100));
}

__global__ void init_vector_array(jek::Vec3f* dev_a)
{
    uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= 10000000) return;

    dev_a[id] = jek::Vec3f(jek::rand_float(-100,100), jek::rand_float(-100, 100), jek::rand_float(-100, 100));
}

__global__ void init_vector_array2(float* x, float* y, float* z)
{
    uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= 10000000) return;

    x[id] = jek::rand_float(-100, 100);
    y[id] = jek::rand_float(-100, 100);
    z[id] = jek::rand_float(-100, 100);
}

__global__ void vec_speed_test(jek::Vec3f* dev_a)
{
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= 10000000-3) return;
    //dev_a[id] = dev_a[id] * jek::Matrix4x4f(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    auto mat = jek::Matrix4x4f(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    jek::Vec3f x;
    //for (int i = 0; i < 10000000-3; i++)
    //{
        x = x + jek::Vec3f(jek::Vec4f(dev_a[i], 1.f)*mat);
        x = x + jek::Vec3f(jek::Vec4f(dev_a[i+1], 1.f)*mat);
        x = x + jek::Vec3f(jek::Vec4f(dev_a[i+2], 1.f)*mat);
    //}
    dev_a[0] = x;
}

__global__ void vec_speed_test2(float* x, float* y, float* z)
{
    //uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
    //if (id >= 10000000-3) return;
    //dev_a[id] = dev_a[id] * jek::Matrix4x4f(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    auto mat = jek::Matrix4x4f(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    for (int i = 0; i < 10000000; i++)
    {
        x[i] = x[i] * jek::M_PI;
        y[i] = y[i] * jek::M_PI;
        z[i] = z[i] * jek::M_PI;
    }
}

int main()
{
    auto p = jek::normalize(jek::Vec4f(2));

    jek::Vec3f* dev_a;

    float* x;
    float* y;
    float* z;

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void**)&dev_a, 10000000 * sizeof(jek::Vec3f));

    cudaStatus = cudaMalloc((void**)&x, 10000000 * sizeof(float));
    cudaStatus = cudaMalloc((void**)&y, 10000000 * sizeof(float));
    cudaStatus = cudaMalloc((void**)&z, 10000000 * sizeof(float));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    uint32_t block_size = 256;
    uint32_t num_blocks = (10000000 + block_size - 1) / block_size;
    init_vector_array <<<num_blocks, block_size >>> (dev_a);
    //init_vector_array2 <<<num_blocks, block_size >>> (x,y,z);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    std::cerr << "\t gathering triangle information for BVH construction: ";
    clock_t start, stop;
    start = clock();

    for (int i = 0; i < 10000; i++)
    {
        block_size = 256;
        num_blocks = (10000000 + block_size - 1) / block_size;
        vec_speed_test << <num_blocks, block_size >> > (dev_a);
        //vec_speed_test2 << <num_blocks, block_size >> > (x,y,z);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }
        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        }
    }
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << timer_seconds << "s" << std::endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<32, 32>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
