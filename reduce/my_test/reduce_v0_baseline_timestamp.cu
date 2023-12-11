#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

double CpuTimeSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec+(double)tp.tv_usec*1e-6);   
}

__global__ void reduce0(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(out[i]!=res[i])
            return false;
    }
    return true;
}

int main(){
    const int N=32*1024*1024;
    float *a=(float *)malloc(N*sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a,N*sizeof(float));

    int block_num=N/THREAD_PER_BLOCK;
    float *out=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float)); // 每个 block 规约输出的数值
    float *d_out; // device 侧面
    cudaMalloc((void **)&d_out,(N/THREAD_PER_BLOCK)*sizeof(float)); // 每个 block reduce 输出的数值
    float *res=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));

    for(int i=0;i<N;i++){
        a[i]=1;
    }

    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<THREAD_PER_BLOCK;j++){
            cur+=a[i*THREAD_PER_BLOCK+j];
        }
        res[i]=cur; // cpu 测进行 reduce 操作 baseline 用于验证结果
    }


    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 Grid( N/THREAD_PER_BLOCK,1); // 一维 的 grid ->blockDim.x
    dim3 Block( THREAD_PER_BLOCK,1);
    // Use CPU time to capture the duration
    double cpuStart,cpuElaps, bandwidth, Cubandwidth;
   
    // Use Cuda event to capture the duration
    cudaEvent_t custart, custop;
    cudaEventCreate(&custart);
    cudaEventCreate(&custop);

    cpuStart = CpuTimeSecond();

    cudaEventRecord(custart, 0);

    reduce0<<<Grid,Block>>>(d_a,d_out);
    //cudaDeviceSynchronize();

    cudaEventRecord(custop, 0);
    cudaEventSynchronize(custop);

    float CudaElaps;
    cudaEventElapsedTime(&CudaElaps, custart, custop);

    cpuElaps = CpuTimeSecond() - cpuStart;  // cpu time

    bandwidth = N*sizeof(float) / cpuElaps / 1024 / 1024 / 1024;
    Cubandwidth = N*sizeof(float) / CudaElaps / 1024 / 1024 / 1024 * 1e3;

    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,block_num)) {
        printf("the ans is right\n");
        printf("Kernel execution CPU time %.6f ms BW: %.2f GB/s\n", cpuElaps*1e3, bandwidth);
        printf("Kernel execution GPU time %.6f ms BW: %.2f GB/s\n", CudaElaps, Cubandwidth);
    }
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_out);
}

