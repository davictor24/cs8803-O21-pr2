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

// How many streams to use for the actual sorting
#define NUM_STREAMS 8

// Block size for shared memory sorting (global memory sorting uses a fraction
// of this)
#define BLOCK_SIZE 1024

// Maximum number of steps global memory sorting should do in one go
#define MAX_GLOBAL_STEPS 3

// Implement your GPU device kernel(s) here (e.g., the bitonic sort kernel).

// Comparator for bitonic sorting
__device__ void compareExchange(DTYPE& a, DTYPE& b, bool ascending) {
  const DTYPE min_ = min(a, b);
  const DTYPE max_ = max(a, b);
  a = ascending ? min_ : max_;
  b = ascending ? max_ : min_;
}

// Runs as many stages as possible that would fit in shared memory.
// Instead of having an if-statement in the loop to decide which thread performs
// the swap, the body of the loop is branchless to avoid warp divergence.
//
// `blockOffset` refers to the total number of blocks in previous streams.
//
// This (and other kernel implementations, plus `performBitonicSort`) follow the
// algorithm in the project info document.
__global__ void bitonicSortInitialShared(DTYPE* __restrict__ arr,
                                         int blockOffset) {
  __shared__ DTYPE shared[BLOCK_SIZE];
  // Dummy array swapped by the thread not responsible for swapping the array
  // elements (to maintain a branchless implementation)
  __shared__ DTYPE dummy[2];

  const int globalIdx = threadIdx.x + (blockIdx.x + blockOffset) * BLOCK_SIZE;
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

// Runs the later steps for a stage that fit into shared memory.
// Similar to `bitonicSortInitialShared`, the body of the loop is also
// branchless.
__global__ void bitonicSortShared(DTYPE* __restrict__ arr, int stage,
                                  int blockOffset) {
  __shared__ DTYPE shared[BLOCK_SIZE];
  __shared__ DTYPE dummy[2];

  const int globalIdx = threadIdx.x + (blockIdx.x + blockOffset) * BLOCK_SIZE;
  shared[threadIdx.x] = arr[globalIdx];

  __syncthreads();

  for (int j = __builtin_ctz(BLOCK_SIZE) - 1; j >= 0; j--) {
    const int partner = threadIdx.x ^ (1 << j);

    const bool isActive = partner > threadIdx.x;
    DTYPE& a = isActive ? shared[threadIdx.x] : dummy[0];
    DTYPE& b = isActive ? shared[partner] : dummy[1];

    const bool ascending = ((globalIdx & (1 << stage)) == 0);
    compareExchange(a, b, ascending);

    __syncthreads();
  }

  arr[globalIdx] = shared[threadIdx.x];
}

// Global memory sorting for steps that don't fit into shared memory.
// `handleSteps` indicate how many steps for the given stage should be done.
__global__ void bitonicSortGlobal(DTYPE* __restrict__ arr, int stage, int step,
                                  int blockOffset, int handleSteps) {
  // Distance between global memory accesses
  const int stride = 1 << (step - handleSteps + 1);

  // Global identifier for thread
  const int tid = threadIdx.x + (blockIdx.x + blockOffset) * blockDim.x;

  // Bitonic sort group (block of interconnected indices)
  const int group = tid >> (step - handleSteps + 1);

  // Offset within group for thread
  const int offset = tid & (stride - 1);

  // First index within group to be handled by thread
  const int idx0 = (group << (step + 1)) + offset;

  const bool ascending = ((idx0 & (1 << stage)) == 0);

  if (handleSteps == 3) {
    const int idx1 = idx0 + stride;
    const int idx2 = idx1 + stride;
    const int idx3 = idx2 + stride;
    const int idx4 = idx3 + stride;
    const int idx5 = idx4 + stride;
    const int idx6 = idx5 + stride;
    const int idx7 = idx6 + stride;

    // Copy once to registers
    DTYPE value0 = arr[idx0];
    DTYPE value1 = arr[idx1];
    DTYPE value2 = arr[idx2];
    DTYPE value3 = arr[idx3];
    DTYPE value4 = arr[idx4];
    DTYPE value5 = arr[idx5];
    DTYPE value6 = arr[idx6];
    DTYPE value7 = arr[idx7];

    compareExchange(value0, value4, ascending);
    compareExchange(value1, value5, ascending);
    compareExchange(value2, value6, ascending);
    compareExchange(value3, value7, ascending);

    compareExchange(value0, value2, ascending);
    compareExchange(value1, value3, ascending);
    compareExchange(value4, value6, ascending);
    compareExchange(value5, value7, ascending);

    compareExchange(value0, value1, ascending);
    compareExchange(value2, value3, ascending);
    compareExchange(value4, value5, ascending);
    compareExchange(value6, value7, ascending);

    // Copy back to global memory
    arr[idx0] = value0;
    arr[idx1] = value1;
    arr[idx2] = value2;
    arr[idx3] = value3;
    arr[idx4] = value4;
    arr[idx5] = value5;
    arr[idx6] = value6;
    arr[idx7] = value7;
  } else if (handleSteps == 2) {
    const int idx1 = idx0 + stride;
    const int idx2 = idx1 + stride;
    const int idx3 = idx2 + stride;

    DTYPE value0 = arr[idx0];
    DTYPE value1 = arr[idx1];
    DTYPE value2 = arr[idx2];
    DTYPE value3 = arr[idx3];

    compareExchange(value0, value2, ascending);
    compareExchange(value1, value3, ascending);

    compareExchange(value0, value1, ascending);
    compareExchange(value2, value3, ascending);

    arr[idx0] = value0;
    arr[idx1] = value1;
    arr[idx2] = value2;
    arr[idx3] = value3;
  } else {
    const int idx1 = idx0 + stride;

    DTYPE value0 = arr[idx0];
    DTYPE value1 = arr[idx1];

    compareExchange(value0, value1, ascending);

    arr[idx0] = value0;
    arr[idx1] = value1;
  }
}

void performBitonicSort(DTYPE* __restrict__ arrGpu,
                        std::vector<cudaStream_t>& streams, int N) {
  const int logN = __builtin_ctz(N);
  const int logBlockSize = __builtin_ctz(BLOCK_SIZE);
  const int logNumStreams = __builtin_ctz(NUM_STREAMS);
  // log(number of elements in a stream)
  const int logStreamSize = logN - logNumStreams;

  const int totalNumBlocks = N / BLOCK_SIZE;
  const int numBlocksPerStream = totalNumBlocks / NUM_STREAMS;

  // Events for enforcing stream dependencies
  std::vector<cudaEvent_t> events(NUM_STREAMS);
  for (int s = 0; s < NUM_STREAMS; s++) {
    cudaEventCreate(&events[s]);
  }

  // Initial shared memory sorting
  for (int s = 0; s < NUM_STREAMS; s++) {
    int blockOffset = s * numBlocksPerStream;
    bitonicSortInitialShared<<<numBlocksPerStream, BLOCK_SIZE, 0, streams[s]>>>(
        arrGpu, blockOffset);
  }

  for (int stage = logBlockSize + 1; stage <= logN; stage++) {
    int step = stage - 1;

    // Global memory sorting for initial steps in stage
    while (step >= logBlockSize) {
      // Number of steps to run at the same time
      int handleSteps = std::min(MAX_GLOBAL_STEPS, step - logBlockSize + 1);

      // Reduce number of threads in a block so that each thread handles
      // 2^handleSteps elements
      int blockSize = BLOCK_SIZE >> handleSteps;

      // Does this step involve blocks in different streams?
      bool crossStream = (step >= logStreamSize);

      if (crossStream) {  // Yes: Enforce stream dependencies
        // How many streams in a dependent stream group
        const int streamsPerGroup = 1 << (step + 1 - logStreamSize);

        // How many blocks in a dependent stream group
        const int blocksPerGroup = streamsPerGroup * numBlocksPerStream;

        // How many stream groups
        const int numGroups = NUM_STREAMS / streamsPerGroup;

        for (int g = 0; g < numGroups; g++) {
          const int groupStart = g * streamsPerGroup;
          // Join all streams within group to leader stream
          cudaStream_t& leaderStream = streams[groupStart];

          for (int s = 0; s < streamsPerGroup; s++) {
            const int streamIdx = groupStart + s;
            cudaEventRecord(events[streamIdx], streams[streamIdx]);
          }

          // Leader waits for all streams within group
          for (int s = 0; s < streamsPerGroup; s++) {
            const int streamIdx = groupStart + s;
            cudaStreamWaitEvent(leaderStream, events[streamIdx], 0);
          }

          // Run global memory sorting for the group on leader stream
          const int blockOffset = groupStart * numBlocksPerStream;
          bitonicSortGlobal<<<blocksPerGroup, blockSize, 0, leaderStream>>>(
              arrGpu, stage, step, blockOffset, handleSteps);

          // All streams within group wait for leader to be done before
          // continuing
          cudaEventRecord(events[groupStart], leaderStream);
          for (int s = 1; s < streamsPerGroup; s++) {
            const int followerIdx = groupStart + s;
            cudaStreamWaitEvent(streams[followerIdx], events[groupStart], 0);
          }
        }
      } else {  // No: Run global memory sorting independently per stream
        for (int s = 0; s < NUM_STREAMS; s++) {
          int blockOffset = s * numBlocksPerStream;
          bitonicSortGlobal<<<numBlocksPerStream, blockSize, 0, streams[s]>>>(
              arrGpu, stage, step, blockOffset, handleSteps);
        }
      }

      step -= handleSteps;
    }

    // Shared memory sorting for remaining steps in stage
    for (int s = 0; s < NUM_STREAMS; s++) {
      int blockOffset = s * numBlocksPerStream;
      bitonicSortShared<<<numBlocksPerStream, BLOCK_SIZE, 0, streams[s]>>>(
          arrGpu, stage, blockOffset);
    }
  }

  for (int s = 0; s < NUM_STREAMS; s++) {
    cudaEventDestroy(events[s]);
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

  std::vector<cudaStream_t> streams(NUM_STREAMS);
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // Input padding is done in a separate stream while input itself is being
  // copied to device.
  cudaStream_t paddingStream;
  cudaStreamCreate(&paddingStream);
  cudaEvent_t paddingCompleteEvent;
  cudaEventCreate(&paddingCompleteEvent);

  DTYPE* arrGpu;
  // All streams have at least one full block
  int N = BLOCK_SIZE * NUM_STREAMS;
  // Round up `size` to the nearest power of 2
  while (N < size) {
    N <<= 1;
  }
  cudaMalloc((void**)&arrGpu, N * sizeof(DTYPE));

  const int paddingLength = N - size;
  if (paddingLength > 0) {
    // Pad with zeros for simplicity - memset cannot be used with INT_MAX.
    // This works because all elements >= 0.
    cudaMemsetAsync(arrGpu + size, 0, paddingLength * sizeof(DTYPE),
                    paddingStream);
    cudaEventRecord(paddingCompleteEvent, paddingStream);
  }

  // Transfer data (arrCpu) to device
  const int chunkSize = N / NUM_STREAMS;
  int copied = 0;
  for (int i = 0; i < NUM_STREAMS && copied < size; i++) {
    const int copySize = std::min(chunkSize, size - copied);
    // `cudaHostRegister` takes too long if done on the entire buffer at once.
    // Register one chunk of the buffer at a time, while asynchronous work is
    // being done on other streams
    cudaHostRegister(arrCpu + copied, copySize * sizeof(DTYPE),
                     cudaHostRegisterDefault);
    cudaMemcpyAsync(arrGpu + copied, arrCpu + copied, copySize * sizeof(DTYPE),
                    cudaMemcpyHostToDevice, streams[i]);
    copied += copySize;
  }

  if (paddingLength > 0) {
    // Wait for padding to complete (if it hasn't already) before continuing.
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

  // Allocate and register buffer for result while asynchronous work is being
  // done on the GPU.
  DTYPE* arrSortedGpu = (DTYPE*)malloc(size * sizeof(DTYPE));
  cudaHostRegister(arrSortedGpu, size * sizeof(DTYPE), cudaHostRegisterDefault);

  /* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpuTime, start, stop);

  cudaEventRecord(start);

  /* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

  // Transfer sorted data back to host (copied to arrSortedGpu)
  copied = 0;
  // Since we padded with zeros, copy from the last index.
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

  cudaFreeAsync(arrGpu, 0);

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