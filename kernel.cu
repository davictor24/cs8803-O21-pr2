#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <ctime>
// ==== DO NOT MODIFY CODE ABOVE THIS LINE ====

#define DTYPE int
// Add any additional #include headers or helper macros needed
#include <vector>

#define NUM_STREAMS 1
#define BLOCK_SIZE 256

// Implement your GPU device kernel(s) here (e.g., the bitonic sort kernel).

__device__ void compareExchange(DTYPE& a, DTYPE& b, bool ascending) {
  const DTYPE min = (a < b) ? a : b;
  const DTYPE max = (a >= b) ? a : b;
  a = ascending ? min : max;
  b = ascending ? max : min;
}

__global__ void bitonicSortInitialShared(DTYPE* arr) {
  __shared__ DTYPE shared[BLOCK_SIZE];
  __shared__ DTYPE dummy[2];

  const int globalIdx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  shared[threadIdx.x] = arr[globalIdx];

  __syncthreads();

  for (int i = 1; i <= __builtin_ctz(BLOCK_SIZE); i++) {
    for (int j = i - 1; j >= 0; j--) {
      const int partner = threadIdx.x ^ (1 << j);
      const bool isActive = partner > threadIdx.x;
      DTYPE& a = isActive ? shared[threadIdx.x] : dummy[0];
      DTYPE& b = isActive ? shared[partner] : dummy[1];
      const bool ascending = ((globalIdx & (1 << i)) == 0);
      compareExchange(a, b, ascending);

      __syncthreads();
    }
  }

  arr[globalIdx] = shared[threadIdx.x];
}

__global__ void bitonicSortShared(DTYPE* arr, int stage) {
  __shared__ DTYPE shared[BLOCK_SIZE];

  const int globalIdx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  shared[threadIdx.x] = arr[globalIdx];

  __syncthreads();

  for (int j = __builtin_ctz(BLOCK_SIZE) - 1; j >= 0; j--) {
    const int partner = threadIdx.x ^ (1 << j);
    if (partner > threadIdx.x) {
      const bool ascending = ((globalIdx & (1 << stage)) == 0);
      compareExchange(shared[threadIdx.x], shared[partner], ascending);
    }

    __syncthreads();
  }

  arr[globalIdx] = shared[threadIdx.x];
}

__global__ void bitonicSortGlobal(DTYPE* arr, int stage, int step) {
  const int globalIdx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  const int partner = globalIdx ^ (1 << step);

  if (partner > globalIdx) {
    const bool ascending = ((globalIdx & (1 << stage)) == 0);
    compareExchange(arr[globalIdx], arr[partner], ascending);
  }
}

void performBitonicSort(DTYPE* arrGpu, std::vector<cudaStream_t>& streams,
                        int N) {
  // TODO: Support multiple streams
  cudaStream_t stream = streams[0];

  const int logN = __builtin_ctz(N);
  const int logBlockN = __builtin_ctz(BLOCK_SIZE);
  const int numGrids = N / BLOCK_SIZE;

  bitonicSortInitialShared<<<numGrids, BLOCK_SIZE, 0, stream>>>(arrGpu);

  for (int i = logBlockN + 1; i <= logN; i++) {
    for (int j = i - 1; j >= 0; j--) {
      if (j >= logBlockN) {
        bitonicSortGlobal<<<numGrids, BLOCK_SIZE, 0, stream>>>(arrGpu, i, j);
      } else {
        bitonicSortShared<<<numGrids, BLOCK_SIZE, 0, stream>>>(arrGpu, i);
        break;
      }
    }
  }
}

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s <array_size>\n", argv[0]);
    return 1;
  }

  int size = atoi(argv[1]);

  srand(time(NULL));

  DTYPE* arrCpu = (DTYPE*)malloc(size * sizeof(DTYPE));

  for (int i = 0; i < size; i++) {
    arrCpu[i] = rand() % 1000;
  }

  float gpuTime, h2dTime, d2hTime, cpuTime = 0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  /* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

  cudaHostRegister(arrCpu, size * sizeof(DTYPE), cudaHostRegisterDefault);

  std::vector<cudaStream_t> streams(NUM_STREAMS);
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
  }

  cudaStream_t paddingStream;
  cudaStreamCreate(&paddingStream);
  cudaEvent_t paddingCompleteEvent;
  cudaEventCreate(&paddingCompleteEvent);

  // arCpu contains the input random array
  // arrSortedGpu should contain the sorted array copied from GPU to CPU
  DTYPE* arrSortedGpu = (DTYPE*)malloc(size * sizeof(DTYPE));

  DTYPE* arrGpu;

  int N = BLOCK_SIZE;
  while (N < size) {
    N <<= 1;
  }
  cudaMalloc((void**)&arrGpu, N * sizeof(DTYPE));

  const int paddingLength = N - size;
  if (paddingLength > 0) {
    cudaMemsetAsync(arrGpu + size, 0, paddingLength * sizeof(DTYPE),
                    paddingStream);
    cudaEventRecord(paddingCompleteEvent, paddingStream);
  }

  // Transfer data (arrCpu) to device
  const int chunkSize = N / NUM_STREAMS;
  int copied = 0;
  for (int i = 0; i < NUM_STREAMS && copied < size; i++) {
    const int copySize = std::min(chunkSize, size - copied);
    cudaMemcpyAsync(arrGpu + copied, arrCpu + copied, copySize * sizeof(DTYPE),
                    cudaMemcpyHostToDevice, streams[i]);
    copied += copySize;
  }

  if (paddingLength > 0) {
    for (int i = NUM_STREAMS / 2; i < NUM_STREAMS; i++) {
      cudaStreamWaitEvent(streams[i], paddingCompleteEvent, 0);
    }
  }

  /* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&h2dTime, start, stop);

  cudaEventRecord(start);

  /* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

  // Perform bitonic sort on GPU
  performBitonicSort(arrGpu, streams, N);

  cudaHostRegister(arrSortedGpu, size * sizeof(DTYPE), cudaHostRegisterDefault);

  /* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpuTime, start, stop);

  cudaEventRecord(start);

  /* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

  // Transfer sorted data back to host (copied to arrSortedGpu)
  copied = 0;
  for (int i = NUM_STREAMS - 1; i >= 0 && copied < size; i--) {
    const int copySize = std::min(chunkSize, size - copied);
    const int destOffset = size - copied - copySize;
    const int srcOffset = N - copied - copySize;
    cudaMemcpyAsync(arrSortedGpu + destOffset, arrGpu + srcOffset,
                    copySize * sizeof(DTYPE), cudaMemcpyDeviceToHost,
                    streams[i]);
    copied += copySize;
  }

  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamDestroy(streams[i]);
  }
  cudaStreamDestroy(paddingStream);
  cudaEventDestroy(paddingCompleteEvent);

  // Not a requirement for the project
  // (https://edstem.org/us/courses/81715/discussion/6897777?comment=16332533)
  // cudaHostUnregister(arrCpu);
  // cudaHostUnregister(arrSortedGpu);

  cudaFree(arrGpu);

  /* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&d2hTime, start, stop);

  auto startTime = std::chrono::high_resolution_clock::now();

  // CPU sort for performance comparison
  std::sort(arrCpu, arrCpu + size);

  auto endTime = std::chrono::high_resolution_clock::now();
  cpuTime =
      std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime)
          .count();
  cpuTime = cpuTime / 1000;

  int match = 1;
  for (int i = 0; i < size; i++) {
    if (arrSortedGpu[i] != arrCpu[i]) {
      match = 0;
      break;
    }
  }

  free(arrCpu);
  free(arrSortedGpu);

  if (match)
    printf("\033[1;32mFUNCTIONAL SUCCESS\n\033[0m");
  else {
    printf("\033[1;31mFUNCTIONCAL FAIL\n\033[0m");
    return 0;
  }

  printf("\033[1;34mArray size         :\033[0m %d\n", size);
  printf("\033[1;34mCPU Sort Time (ms) :\033[0m %f\n", cpuTime);
  float gpuTotalTime = h2dTime + gpuTime + d2hTime;
  int speedup = (gpuTotalTime > cpuTime) ? (gpuTotalTime / cpuTime)
                                         : (cpuTime / gpuTotalTime);
  float meps = size / (gpuTotalTime * 0.001) / 1e6;
  printf("\033[1;34mGPU Sort Time (ms) :\033[0m %f\n", gpuTotalTime);
  printf(
      "\033[1;34mGPU Sort Speed     :\033[0m %f million elements per second\n",
      meps);
  if (gpuTotalTime < cpuTime) {
    printf("\033[1;32mPERF PASSING\n\033[0m");
    printf(
        "\033[1;34mGPU Sort is \033[1;32m %dx \033[1;34mfaster than CPU "
        "!!!\033[0m\n",
        speedup);
    printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
    printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
    printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
  } else {
    printf("\033[1;31mPERF FAILING\n\033[0m");
    printf(
        "\033[1;34mGPU Sort is \033[1;31m%dx \033[1;34mslower than CPU, "
        "optimize further!\n",
        speedup);
    printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
    printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
    printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
    return 0;
  }

  return 0;
}
