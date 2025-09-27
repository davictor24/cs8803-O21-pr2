#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <ctime>
// ==== DO NOT MODIFY CODE ABOVE THIS LINE ====

#define DTYPE int
// Add any additional #include headers or helper macros needed
#define NUM_STREAMS 4
#define COMPARATOR_WIDTH 4
#define BLOCK_SIZE 1024
#define SHARED_SIZE (BLOCK_SIZE * COMPARATOR_WIDTH)

// Implement your GPU device kernel(s) here (e.g., the bitonic sort kernel).

// Branchless compare-exchange
__device__ void compareExchange(DTYPE& a, DTYPE& b, bool ascending) {
  DTYPE lessMask = -((a - b) < 0);
  DTYPE min = (lessMask & a) | (~lessMask & b);
  DTYPE max = (lessMask & b) | (~lessMask & a);

  DTYPE ascMask = -(ascending);
  a = (ascMask & min) | (~ascMask & max);
  b = (ascMask & max) | (~ascMask & min);
}

// Hand-coded bitonic sorting network for 2 elements
__device__ void bitonicSortingNetwork2(DTYPE* a, DTYPE* b, bool ascending) {
  compareExchange(a[0], b[0], ascending);
}

// Hand-coded bitonic sorting network for 4 elements
__device__ void bitonicSortingNetwork4(DTYPE* a, DTYPE* b, bool ascending) {
  bitonicSortingNetwork2(&a[0], &a[1], ascending);
  bitonicSortingNetwork2(&b[0], &b[1], !ascending);

  compareExchange(a[0], b[0], ascending);
  compareExchange(a[1], b[1], ascending);

  bitonicSortingNetwork2(&a[0], &a[1], ascending);
  bitonicSortingNetwork2(&b[0], &b[1], ascending);
}

// Hand-coded bitonic sorting network for 8 elements
__device__ void bitonicSortingNetwork8(DTYPE* a, DTYPE* b, bool ascending) {
  bitonicSortingNetwork4(&a[0], &a[2], ascending);
  bitonicSortingNetwork4(&b[0], &b[2], !ascending);

  compareExchange(a[0], b[0], ascending);
  compareExchange(a[1], b[1], ascending);
  compareExchange(a[2], b[2], ascending);
  compareExchange(a[3], b[3], ascending);

  bitonicSortingNetwork4(&a[0], &a[2], ascending);
  bitonicSortingNetwork4(&b[0], &b[2], ascending);
}

// Compare-exchange for COMPARATOR_WIDTH elements
__device__ void compareExchangeBlock(DTYPE* a, DTYPE* b, bool ascending) {
#if COMPARATOR_WIDTH == 1
  bitonicSortingNetwork2(a, b, ascending);
#elif COMPARATOR_WIDTH == 2
  bitonicSortingNetwork4(a, b, ascending);
#elif COMPARATOR_WIDTH == 4
  bitonicSortingNetwork8(a, b, ascending);
#else
#error "Provided COMPARATOR_WIDTH not supported."
#endif
}

__global__ void bitonicSortInitialShared(DTYPE* arr) {
  __shared__ DTYPE shared[SHARED_SIZE];

  int k = threadIdx.x;
  int sharedIdx = k * COMPARATOR_WIDTH;
  int globalIdx = blockIdx.x * SHARED_SIZE + sharedIdx;

  for (int i = 0; i < COMPARATOR_WIDTH; i++) {
    shared[sharedIdx + i] = arr[globalIdx + i];
  }

  __syncthreads();

  for (int i = 1; i <= __builtin_ctz(BLOCK_SIZE); i++) {
    for (int j = i - 1; j >= 0; j--) {
      int partner = k ^ (1 << j);
      if (partner > k) {
        bool ascending = ((k & (1 << i)) == 0);
        compareExchangeBlock(&shared[sharedIdx],
                             &shared[partner * COMPARATOR_WIDTH], ascending);
      }

      __syncthreads();
    }
  }

  for (int i = 0; i < COMPARATOR_WIDTH; i++) {
    arr[globalIdx + i] = shared[sharedIdx + i];
  }
}

__global__ void bitonicSortShared(DTYPE* arr, int stage) {
  __shared__ DTYPE shared[SHARED_SIZE];

  int k = threadIdx.x;
  int sharedIdx = k * COMPARATOR_WIDTH;
  int globalIdx = blockIdx.x * SHARED_SIZE + sharedIdx;

  for (int i = 0; i < COMPARATOR_WIDTH; i++) {
    shared[sharedIdx + i] = arr[globalIdx + i];
  }

  __syncthreads();

  for (int j = __builtin_ctz(BLOCK_SIZE) - 1; j >= 0; j--) {
    int partner = k ^ (1 << j);
    if (partner > k) {
      bool ascending = ((k & (1 << stage)) == 0);
      compareExchangeBlock(&shared[sharedIdx],
                           &shared[partner * COMPARATOR_WIDTH], ascending);
    }

    __syncthreads();
  }

  for (int i = 0; i < COMPARATOR_WIDTH; i++) {
    arr[globalIdx + i] = shared[sharedIdx + i];
  }
}

__global__ void bitonicSortGlobal(DTYPE* arr, int stage, int step) {
  int k = threadIdx.x + blockIdx.x * blockDim.x;
  int partner = k ^ (1 << step);

  if (partner > k) {
    bool ascending = ((k & (1 << stage)) == 0);
    compareExchangeBlock(&arr[k * COMPARATOR_WIDTH],
                         &arr[partner * COMPARATOR_WIDTH], ascending);
  }
}

void performBitonicSort(DTYPE* arrGpu, std::vector<cudaStream_t>& streams,
                        int N, int logN) {
  // TODO: bitonicSort<<<grid, block>>>(arrGpu);

  int i = 1;

  // Invoke bitonicSortInitialShared and increment i appropriately

  while (i <= logN) {
    int j = i - 1;
    while (j >= 0) {
      // Figure out whether to call bitonicSortGlobal or bitonicSortShared
      // If bitonicSortGlobal, decrement j by (at most) log(COMPARATOR_WIDTH)
      // If bitonicSortShared, break

      // j--;
    }
    i++;
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

  // arCpu contains the input random array
  // arrSortedGpu should contain the sorted array copied from GPU to CPU
  DTYPE* arrSortedGpu = (DTYPE*)malloc(size * sizeof(DTYPE));

  DTYPE* arrGpu;

  int N = 1;
  int logN = 0;
  while (N < size) {
    N <<= 1;
    logN++;
  }
  cudaMalloc((void**)&arrGpu, N * sizeof(DTYPE));
  cudaMemsetD32Async(arrGpu + size, INT_MAX, N - size, 0);

  // Transfer data (arrCpu) to device
  int chunkSize = N / NUM_STREAMS;
  int copied = 0;
  for (int i = 0; i < NUM_STREAMS && copied < size; i++) {
    int copySize = std::min(chunkSize, size - copied);
    cudaMemcpyAsync(arrGpu + copied, arrCpu + copied, copySize * sizeof(DTYPE),
                    cudaMemcpyHostToDevice, streams[i]);
    copied += chunkSize;
  }

  /* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&h2dTime, start, stop);

  cudaEventRecord(start);

  /* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

  // Perform bitonic sort on GPU
  performBitonicSort(arrGpu, streams, N, logN);

  cudaHostRegister(arrSortedGpu, size * sizeof(DTYPE), cudaHostRegisterDefault);

  /* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpuTime, start, stop);

  cudaEventRecord(start);

  /* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

  // Transfer sorted data back to host (copied to arrSortedGpu)
  copied = 0;
  for (int i = 0; i < NUM_STREAMS && copied < size; i++) {
    int copySize = std::min(chunkSize, size - copied);
    cudaMemcpyAsync(arrSortedGpu + copied, arrGpu + copied,
                    copySize * sizeof(DTYPE), cudaMemcpyDeviceToHost,
                    streams[i]);
    copied += chunkSize;
  }

  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamDestroy(&streams[i]);
  }

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
