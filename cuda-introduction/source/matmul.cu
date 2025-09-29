#include "common.h"

#include <device_launch_parameters.h>

#include <cmath>
#include <iostream>
#include <random>

constexpr int TILE_WIDTH = 16;

// TODO 10: Implement the matrix multiplication kernel
__global__ void matrixMultiplicationNaive(float* const matrixP, const float* const matrixM, const float* const matrixN,
                                          const unsigned sizeMX, const unsigned sizeNY, const unsigned sizeXY)
{
    // TODO 10a: Compute the P matrix global index for each thread along x and y dimentions.
    // Remember that each thread of the kernel computes the result of 1 unique element of P
    __shared__ float sM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sN[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; 		int by = blockIdx.y;
    int tx = threadIdx.x;		int ty = threadIdx.y;
    float dot = 0.0;
    unsigned pc = bx * blockDim.x + tx; // global column (x)
    unsigned pr = by * blockDim.y + ty; // global row (y)

    // TODO 10b: Check if px or py are out of bounds. If they are, return.
    // if (pr >= sizeNY || pc >= sizeMX) return;
    // TODO 10c: Compute the dot product for the P element in each thread
    for (unsigned m = 0; m < (sizeXY + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        const unsigned kN = m * TILE_WIDTH + tx; // N column (k)
        const unsigned kM = m * TILE_WIDTH + ty; // M row (k)
        sN[ty][tx] = (pr < sizeNY && kN < sizeXY) ? matrixN[pr * sizeXY + kN] : 0.0f;
        sM[ty][tx] = (kM < sizeXY && pc < sizeMX) ? matrixM[kM * sizeMX + pc] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            dot += sN[ty][k] * sM[k][tx];
        }
        __syncthreads();
    }
    // This loop will be the same as the host loop
    // float dot = 0.0;

    // TODO 10d: Copy dot to P matrix
    // matrixP[] = dot;
    if (pr < sizeNY && pc < sizeMX)
        matrixP[pr * sizeMX + pc] = dot;
}

int main(int argc, char *argv[])
{
    // TODO 1: Initialize sizes. Start with simple like 16x16, then try 32x32.
    // Then try large multiple-block square matrix like 64x64 up to 2048x2048.
    // Then try square, non-power-of-two like 15x15, 33x33, 67x67, 123x123, and 771x771
    // Then try rectangles with powers of two and then non-power-of-two.
    const unsigned sizeMX = 31;
    const unsigned sizeXY = 31;
    const unsigned sizeNY = 31;

    // TODO 2: Allocate host 1D arrays for:
    // matrixM[sizeMX, sizeXY]
    // matrixN[sizeXY, sizeNY]
    // matrixP[sizeMX, sizeNY]
    // matrixPGold[sizeMX, sizeNY]
    float* matrixM = new float[sizeMX * sizeXY];
    float* matrixN = new float[sizeXY * sizeNY];
    float* matrixP = new float[sizeMX * sizeNY];
    float* matrixPGold = new float[sizeMX * sizeNY];

    // LOOK: Setup random number generator and fill host arrays and the scalar a.
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    // Fill matrix M on host
    for (unsigned i = 0; i < sizeMX * sizeXY; i++)
        matrixM[i] = dist(mt);

    // Fill matrix N on host
    for (unsigned i = 0; i < sizeXY * sizeNY; i++)
        matrixN[i] = dist(mt);

    // TODO 3: Compute "gold" reference standard
    // for py -> 0 to sizeNY
    //   for px -> 0 to sizeMX
    //     initialize dot product accumulator
    //     for k -> 0 to sizeXY
    //       dot = m[k, px] * n[py, k]
    //  matrixPGold[py, px] = dot
    // P = N @ M^T, P(sizeNY, sizeMX) = N(sizeNY, sizeXY) * M^T(sizeXY, sizeMX)
    for (unsigned py = 0; py < sizeNY; py++)
    {
        for (unsigned px = 0; px < sizeMX; px++)
        {
            float dot = 0.0f;
            for (unsigned k = 0; k < sizeXY; k++)
            {
                dot += matrixM[k * sizeMX + px] * matrixN[py * sizeXY + k];
            }
            matrixPGold[py * sizeMX + px] = dot;
        }
    }

    // Device arrays
    float *d_matrixM, *d_matrixN, *d_matrixP;

    // TODO 4: Allocate memory on the device for d_matrixM, d_matrixN, d_matrixP.
    CUDA(cudaMalloc((void **)&d_matrixM, sizeMX * sizeXY * sizeof(float)));
    CUDA(cudaMalloc((void **)&d_matrixN, sizeXY * sizeNY * sizeof(float)));
    CUDA(cudaMalloc((void **)&d_matrixP, sizeMX * sizeNY * sizeof(float)));

    // TODO 5: Copy array contents of M and N from the host (CPU) to the device (GPU)
    CUDA(cudaMemcpy(d_matrixM, matrixM, sizeMX * sizeXY * sizeof(float), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d_matrixN, matrixN, sizeXY * sizeNY * sizeof(float), cudaMemcpyHostToDevice));
    CUDA(cudaDeviceSynchronize());

    ////////////////////////////////////////////////////////////
    std::cout << "****************************************************" << std::endl;
    std::cout << "***Matrix Multiplication***" << std::endl;

    // LOOK: Use the clearHostAndDeviceArray function to clear matrixP and d_matrixP
    clearHostAndDeviceArray(matrixP, d_matrixP, sizeMX * sizeNY);

    // TODO 6: Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
    // Calculate number of blocks along X and Y in a 2D CUDA "grid" using divup
    // HINT: The shape of matrices has no impact on launch configuaration
    DIMS dims;
    unsigned BS_X = 16;
    unsigned BS_Y = 16;
    dims.dimBlock = dim3(BS_X, BS_Y, 1);
    unsigned BlockX = divup(sizeMX, BS_X);
    unsigned BlockY = divup(sizeNY, BS_Y);
    dims.dimGrid  = dim3(BlockX, BlockY, 1);
    std::cout << "Launching kernel with " << dims.dimGrid.x << " x " << dims.dimGrid.y << " blocks of "
              << dims.dimBlock.x << " x " << dims.dimBlock.y << " threads" << std::endl;
    // TODO 7: Launch the matrix transpose kernel
    // matrixMultiplicationNaive<<<>>>();
    matrixMultiplicationNaive<<<dims.dimGrid, dims.dimBlock>>>(
        d_matrixP, d_matrixM, d_matrixN, sizeMX, sizeNY, sizeXY);
    // TODO 8: copy the answer back to the host (CPU) from the device (GPU)
    CUDA(cudaMemcpy(matrixP, d_matrixP, sizeMX * sizeNY * sizeof(float), cudaMemcpyDeviceToHost));
    // LOOK: Use compareReferenceAndResult to check the result
    compareReferenceAndResult(matrixPGold, matrixP, sizeMX * sizeNY, 1e-3);

    std::cout << "****************************************************" << std::endl << std::endl;
    ////////////////////////////////////////////////////////////

    // TODO 9: free device memory using cudaFree
    CUDA(cudaFree(d_matrixM));
    CUDA(cudaFree(d_matrixN));
    CUDA(cudaFree(d_matrixP));
    // free host memory
    delete[] matrixM;
    delete[] matrixN;
    delete[] matrixP;
    delete[] matrixPGold;

    // successful program termination
    return 0;
}
