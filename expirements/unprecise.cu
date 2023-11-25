#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cublas_v2.h>
#include <ctime>
#include <curand_kernel.h>
//#include <cooperative_groups.h>
//#include <mex.h>
using namespace std;

// nvcc unprecise.cu -o weight -lcublas -arch=sm_89

// nvcc optCudaV2.cu -o optBLAS -lcublas
// nvcc path C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin\nvcc.exe
// file path C:\Users\Peter Oostema\Documents\school\graphResearch\Mar312020-20200401T020813Z-001\Mar312020\graphEmbed\Aug23Update
//#define PARAM_ARG ssGetSFcnParam(S, 0);
// blockIdx.x * blockDim.x + threadIdx.x;
// nvcc optimizeCuda.cu -o optCuda
// nvprof optCuda
__global__ void transpose(double* N, double* NT, unsigned dimensionality, unsigned n){
    int i = threadIdx.x + blockIdx.y * 1024;
    int j = blockIdx.x;
    if (i < dimensionality){
       NT[i*n + j] = N[i + j*dimensionality];
       //printf("i, j, d, N, NT: %d, %d, %u, %lf, %lf\n", i, j, dimensionality, N[i + j*dimensionality], NT[i*n + j]);
    }
    //printf("threadIdx, threadIdy: %d, %d\n", threadIdx.x, threadIdx.y);

}

// 4 flops
// 4 Gflops
// 43 ms
// 93 Gflops/sec
//
// partial in dist
// 8 flops
// 8 Gflops
// 48 ms
// 166 Gflops/sec
__global__ void findDists(double* currentDists, double* N, double* NT, unsigned n, unsigned dimensionality, double c0){
    int i = threadIdx.x  + blockIdx.y * 1024;
    int j = blockIdx.x;

    if (i < n){
    //double startDist = currentDists[i + j*n];
    double currentDist = 0.0;
    for (int k = 0; k < dimensionality; k++){
        double dist = N[j*dimensionality + k] - NT[i + k*n];
        //currentDists[i + j*n] += dist*dist + 0.000001;
        currentDist += dist*dist;
    }
    currentDist = sqrt(currentDist);
    //double distPreSqrt = currentDists[i + j*n];
    //currentDists[i + j*n] = sqrt(currentDists[i + j*n]);
    //double distPreSqrt = currentDist;
    
    // save partial instead;
    currentDists[i + j*n] = currentDist;
    //currentDists[i + j*n] = (1.0/currentDist) * (c0 / (currentDist*currentDist));
    //double distAfter = currentDists[i + j*n];
    //printf("threadID, blockID, curDist, startDist, distPreSqrt, distAfter, index: %d, %d, %f, %f, %f, %f, %d\n", i, j, currentDists[i + j*n], startDist, distPreSqrt, distAfter, i+j*n);
    }
}

__device__ void pointwiseSQRT(double* currentDists, unsigned n){
    int i = threadIdx.x;
    int j = blockIdx.x;
    currentDists[i + j*n] = sqrt(currentDists[i + j*n]);
}

// 7 flops
// 7 GFlops total
// 77 ms
// 90 Gflops/sec
//
// partial in dist
// 3 flops
// 3 Gflops
// 30 ms
// 100 Gflops/sec
__global__ void NBodyForces(double* N, double* NT, double* V, double* currentDists, unsigned dimensionality, unsigned n, double c0){
    int k = threadIdx.x + blockIdx.y * 1024;
    int i = blockIdx.x;
    if (k < dimensionality){
    for (int j = 0; j < n; j++){
        if (!(i == j)){
            double currentDist = currentDists[i*n + j];
            double partial = (1.0/currentDist) * (c0 / (currentDist*currentDist));
            //double partial = currentDists[i*n + j];
            double force = (((N[k + i*dimensionality] - NT[k*n + j]))*partial);
            V[k + i*dimensionality] += force;
            //if (force != force){
            //   printf("i, j, k, curDist, dif, partal: %d, %d, %d, %f, %f, %f\n", i, j, k, currentDist, N[k + i*dimensionality] - NT[k*n + j], partial);
            //}
        }
    }
    }
}


//            currentDist = norm(y-x);
//            V(:, i) = V(:, i) + ((y - x)/currentDist)*(-c1 * nthroot(currentDist - w, 3));
//            V(:, j) = V(:, j) - ((y - x)/currentDist)*(-c1 * nthroot(currentDist - w, 3));
__global__ void edgeForces(double* W, double* N, double* NT, double* V, double* currentDists, unsigned dimensionality, unsigned n, double c1, double edgeEase){
    int k = threadIdx.x + blockIdx.y * 1024;
    int i = blockIdx.x;
    if (k < dimensionality){
    for (int j = 0; j < n; j++){
        if (!(i == j)){
        if (!(W[i*n + j] != W[i*n + j])){
            double currentDist = currentDists[i*n + j];
                //for (k = 0; k < dimensionality; k++){
                    double relativeDist = currentDist - (((W[i*n + j] - 1) * edgeEase) + 1);
                    //double force = ((N[k + i*dimensionality] - N[k + j*dimensionality])/currentDist) * (c1 * exp(log(fabs(relativeDist))/3.0));
                    double force = ((N[k + i*dimensionality] - N[k + j*dimensionality])/currentDist) * (c1 * pow(fabs(relativeDist),0.33));
                    double scale = fabs(relativeDist) / W[i*n + j];
                    if (scale < 0.4){ // 0.2 for existing tests
                     force *= 0.01; // 0.001 for existing tests
                    }
                     if (relativeDist > 0){
                           V[k + i*dimensionality] -= force;
                           //V[k + j*dimensionality] += force;
                     } else if (relativeDist < 0) {
                           V[k + i*dimensionality] += force;
                           //V[k + j*dimensionality] -= force; 
                     }
                //}
                //printf("curDist, i, j, weight, relDist, force, V[]: %f, %d, %d, %f, %f, %f, %f\n", currentDist, i, j, W[i*n + j], relativeDist, force, V[k + i*dimensionality]);
            // double force = ((N[k + i*dimensionality] - N[k + j*dimensionality])/currentDist) * c1;
            // V[k + i*dimensionality] -= force;
        }
        }
    }
    }
}



__global__
void nBodyIteration(double* N, double* V, unsigned dimensionality, unsigned n, double c0){
   double currentDist = 0;
   //double energy = 0;
   int k = 0;
   int j = 0;
   int i = threadIdx.x;
      //j = blockIdx.x;
      for (j = 0; j < n; j++){//n
        if (i == j){
           k = 0;
        } else {
            // 2 norm
            //double currentDist = 0;
            //unsigned k = 0;
            currentDist = 0;
            for (k = 0; k < dimensionality; k++){
                double dist = N[k + i*dimensionality] - N[k + j*dimensionality];
                currentDist += dist*dist + 0.000001;
            }
            currentDist = sqrt(currentDist);
            //energy += 1.0 / (currentDist*currentDist);
            
            double partial = (1.0/currentDist) * (c0 / (currentDist*currentDist));
            // iterate over dims
            for (k = 0; k < dimensionality; k++){
                double force = (((N[k + i*dimensionality] - N[k + j*dimensionality]))*partial);
                //double force = (((N[k + i*dimensionality] - N[k + j*dimensionality]))) * (c0 / (currentDist*currentDist));
                V[k + i*dimensionality] += force;
                //V[k + j*dimensionality] -= force;
            }
          }
      }
}


void fixWorstEdge(double* N, double* W, unsigned dimensionality, unsigned n){
   double currentDist = 0;
   //double energy = 0;
   int k = 0;
   int j = 0;
   int i = 0;
   //printf("shift start N: %p\n", N);

   double maxDist = 0;
   int maxi = 0;
   int maxj = 0;
           
   for (i = 0; i < n; i++){
      for (j = i+1; j < n; j++){//n
        if (i == j){
           k = 0;
        } else {
           // 2 norm
           //double currentDist = 0;
           //unsigned k = 0;
           currentDist = 0;
           for (k = 0; k < dimensionality; k++){
              double dist = N[k + i*dimensionality] - N[k + j*dimensionality];
              currentDist += dist*dist + 0.00000001;
           }
           currentDist = abs( 1 - sqrt(currentDist));

           if (maxDist < currentDist){
              maxDist = currentDist;
              maxi = i;
              maxj = j;
           }
         }
      }
      //printf("N,k,i: %f %d %d\n", N[k + i*dimensionality], k, i);
   }  
   //printf("shift end N: %p\n", N);
   for (int t = 0; t < 10; t++){
   if (!(W[maxi*n + maxj] != W[maxi*n + maxj])){
     for (k = 0; k < dimensionality; k++){
        double relativeDist = currentDist - W[i*n + j];
        double co = 0.1;
        double force = ((N[k + maxi*dimensionality] - N[k + maxj*dimensionality])/currentDist) * relativeDist * co;
        if (relativeDist > 0){
           N[k + maxi*dimensionality] -= force;
           N[k + maxj*dimensionality] += force;
        } else {
           N[k + maxi*dimensionality] += force;
           N[k + maxj*dimensionality] -= force; 
        }
     }
   }
   }
}
  
        
__global__
void copyToPadded(double* NCopy, double* Ncu, unsigned dp, unsigned d, unsigned n){
   int i = threadIdx.x + blockIdx.x * 32;
   int j = threadIdx.y + blockIdx.y * 32;
//    if (i == 31 && j == 31){
//       printf("here\n");
//    }
   if ((i < d) && (j < n)){
      NCopy[j*dp + i] = Ncu[j*d + i];
   } else if ((i < dp) && (j < n)){
      NCopy[j*dp + i] = 0.0;
   }
}

__global__
void copyToPaddedT(double* NTCopy, double* NTcu, unsigned dp, unsigned d, unsigned n){
   int i = threadIdx.x + blockIdx.x * 32;
   int j = threadIdx.y + blockIdx.y * 32;
   if ((i < n) && (j < d)){
      NTCopy[j*n + i] = NTcu[j*n + i];
   } else if ((j < dp) && (i < n)){
      NTCopy[j*n + i] = 0.0;
   }
}

__global__
void ptwSquare(double* square, double* base, unsigned n, unsigned d){
   int index = threadIdx.x + blockIdx.x * 1024;
   if (index < n*d){
      double baseI = base[index];
      square[index] = baseI * baseI;
   }
//    if (index > 999000){
//       printf("index, val: %d, %f\n", index, base[index]);
//    }
}

__global__
void squareReduce(double* ik, double* N, unsigned n, unsigned d){
   int index = threadIdx.x + blockIdx.x * 1024;
   if (index < n){
      double sum = 0;
      for (int k = 0; k < d; k++){
         sum += N[index*d + k];
      }
//       if ((index == 999) || (index == 998)){
//          printf("ikpartial\n");
//          for (int k = 0; k < d; k++){
//              printf("%d, %f \n", index, N[index*d + k]);
//          } 
//       }
      ik[index] = sum;
//       printf("index, sum: %d, %f \n", index, sum);
   }
}

__global__
void distSums(double* sum, double* ik, double* mul, unsigned n){
   int i = threadIdx.x + blockIdx.x * 32;
   int j = threadIdx.y + blockIdx.y * 32;
   if ((i < n) && (j < n)){
      //const double sameLocationError = 0.000001;
      double result = sqrt(ik[i] + ik[j] + mul[i*n + j]);// + sameLocationError;
      sum[i*n + j] = result;
      //if (i == j){
         //printf("i, ik, mul: %d, %f, %f, %f, %f\n", i, ik[i], ik[j], mul[i*n + j], result);
      //}
   }
}

__global__
void partialComp(double* partials, double* currentDists, double c0, unsigned n){
   int i = threadIdx.x + blockIdx.x * 32;
   int j = threadIdx.y + blockIdx.y * 32;
   if ((i < n) && (j < n)){
      const double distScaleAdjust = 1;
      double dist = currentDists[j*n + i];// * distScaleAdjust + 0.00000001;
      //printf("i,j,dist: %lf, %d, %d\n", dist, i, j);
      //partials[j*n + i] = c0 / (dist*dist*dist);
      partials[j*n + i] = c0 / (dist*dist*dist);
      if (i == j){
         partials[j*n + i] = 0.0;
      }
   }
}

// void nDivByDist(double* NPartials, double* NT, double* currentDists, unsigned n){
//    int i = threadIdx.x + blockIdx.x * 32;
//    int j = threadIdx.y + blockIdx.y * 32;
//    if ((i < n) && (j < n)){
//       NPartials[j*n + i] = ;
//    }
// }

__global__ void NBodyMul(double* N, double* NT, double* V, double* partials, unsigned dimensionality, unsigned n){
    int k = threadIdx.x + blockIdx.y * 1024;
    int i = blockIdx.x;
    if (k < dimensionality){
       double staticNPos = N[k + i*dimensionality];
       double force = 0.0;
       for (int j = 0; j < n; j++){
          if (!(i == j)){
             //force += (staticNPos - NT[k*n + j])*partials[i*n + j];
             force += (staticNPos - N[k + j*dimensionality])*partials[i*n + j];
             //force += (staticNPos - N[k + j*dimensionality])*parShared[j];
             //force += (staticNPos - NShared[kMod + j*32])*partials[i*n + j];
          }
       }
       V[k + i*dimensionality] += force;
    }
}

__global__ void NBodyMul(double* N, double* NT, double* V, double* partials, unsigned dimensionality, unsigned dimensionalitypad, unsigned n){
//    // int k = threadIdx.x + blockIdx.y * 1024;
//    // int i = blockIdx.x;

// //     extern __shared__ double parShared[];
// //     for (int j = (k % 32); j < n; j += 32){
// //        parShared[j] = partials[i*n + j];
// //     }
// //     extern __shared__ double NShared[];
// //     int kMod = k % 32;
// //     for (int j = 0; j < n; j++){
// //        //if (i == 0){
// //           //printf("addr: n: %d %d \n", kMod + j*32, n);
// //        //}
// //        NShared[kMod + j*32] = N[k + j*dimensionality];
// //     }
    

//     if (k < dimensionality){
//        double staticNPos = N[k + i*dimensionality];
//        double force = 0.0;
// //        int jTile = 0;
// //        for (jTile = 128; jTile < n; jTile += 128){
//        for (int j = 0; j < n; j++){
//           if (!(i == j)){
//              //force += (staticNPos - NT[k*n + j])*partials[i*n + j];
//              force += (staticNPos - N[k + j*dimensionality])*partials[i*n + j];
//              //force += (staticNPos - N[k + j*dimensionality])*parShared[j];
//              //force += (staticNPos - NShared[kMod + j*32])*partials[i*n + j];
//           }
//        }
// //           for (int j = jTile-128; j < jTile; j++){
// //             if (!(i == j)){
// //                force += (staticNPos - N[k + j*dimensionality])*partials[i*n + j];
// //             }
// //           }
// //           __syncthreads();
// //        }
// //        for (int j = jTile-128; j < n; j++){
// //           if (!(i == j)){
// //                force += (staticNPos - N[k + j*dimensionality])*partials[i*n + j];
// //           }
// //        }
//        V[k + i*dimensionality] += force;
//     }

   int k = blockIdx.y * 8;
   int i = blockIdx.x * 1024 + threadIdx.x;
   int iTile = threadIdx.x / 8;
   int iTileDim = threadIdx.x % 8;
   __shared__ double NShared[64*8];
   //printf("threadIdx, tidMod: %d, %d \n", threadIdx.x, iTile);
   //printf("i: %d \n", i);
   //printf("k: %d \n", k);
   double staticNPos0 = 0.0;
   double staticNPos1 = 0.0;
   double staticNPos2 = 0.0;
   double staticNPos3 = 0.0;
   double staticNPos4 = 0.0;
   double staticNPos5 = 0.0;
   double staticNPos6 = 0.0;
   double staticNPos7 = 0.0;
   double force0 = 0.0;
   double force1 = 0.0;
   double force2 = 0.0;
   double force3 = 0.0;
   double force4 = 0.0;
   double force5 = 0.0;
   double force6 = 0.0;
   double force7 = 0.0;
   if (i < n){
      staticNPos0 = N[i*dimensionalitypad + k + 0];
      staticNPos1 = N[i*dimensionalitypad + k + 1];
      staticNPos2 = N[i*dimensionalitypad + k + 2];
      staticNPos3 = N[i*dimensionalitypad + k + 3];
      staticNPos4 = N[i*dimensionalitypad + k + 4];
      staticNPos5 = N[i*dimensionalitypad + k + 5];
      staticNPos6 = N[i*dimensionalitypad + k + 6];
      staticNPos7 = N[i*dimensionalitypad + k + 7];
   }
   
   for (int tileIndex = 0; tileIndex < n; tileIndex += 64){
      // tile of 32 nodes, 8 dims at a time
      int tileBoundary = tileIndex + 64;
      int tileWidth = 64;
      if (tileBoundary > n){
         tileWidth = n % 64;
      }
      __syncthreads();
      if (iTile < tileWidth){
         //for (int j = 0; j < 8; j++){
            // TODO check boundry on tile 32 nodes at a time
            //printf("j: %d\n", j);
            // if (N[(threadIdx.x + tileIndex)*dimensionalitypad + (k + j)] != NT[(threadIdx.x + tileIndex) + (k + j)*n]){
            //    printf("i, j, N, NT: %d, %d, %lf, %lf \n", k + j, threadIdx.x + tileIndex, N[(threadIdx.x + tileIndex)*dimensionalitypad + (k + j)], NT[(threadIdx.x + tileIndex) + (k + j)*n]);
            // }
            //NShared[threadIdx.x + j*tileWidth] = NT[(threadIdx.x + tileIndex) + (k + j)*n];
            //NShared[threadIdx.x] = N[(iTile + tileIndex)*dimensionalitypad + (k + iTileDim)];
            //NShared[threadIdx.x] = NT[(iTile + tileIndex) + (k + iTileDim)*n];
            NShared[threadIdx.x] = N[(iTile + tileIndex)*dimensionalitypad + (k + iTileDim)];

            // if (N[(iTile + tileIndex)*dimensionalitypad + (k + iTileDim)] != NT[(iTile + tileIndex) + (k + iTileDim)*n]){
            //    printf("NT, N, tID, i, j, d: %lf, %lf, %d, %d, %d, %d \n", NT[(iTile + tileIndex) + (k + iTileDim)*n], N[(iTile + tileIndex)*dimensionality + (k + iTileDim)], threadIdx.x, (iTile + tileIndex), (k + iTileDim), dimensionality);
            // }
            //printf("Ns, N, tID, tileIndex: %lf, %lf, %d, %d \n", NShared[threadIdx.x], N[(iTile + tileIndex)*dimensionalitypad + (k + iTileDim)], threadIdx.x, tileIndex);
         //}
      }
      //printf("Ns, tID, tIDMod: %lf, %d, %d \n", NShared[threadIdx.x], threadIdx.x, iTile);

      __syncthreads();
      if (i < n){
         for (int j = 0; j < tileWidth; j++){
            // if (NShared[0 + j*8] != N[k + 0 + (tileIndex + j)*dimensionalitypad]){
            //    printf("i, k, Nshared, N 0: %d, %d, %lf, %lf \n", i, k, NShared[0*tileWidth + j], N[k + 0 + (tileIndex + j)*dimensionalitypad]);
            // }
            // if (NShared[1 + j*8] != N[k + 1 + (tileIndex + j)*dimensionalitypad]){
            //    printf("i, k, Nshared, N 1: %d, %d, %lf, %lf \n", i, k, NShared[1*tileWidth + j], N[k + 1 + (tileIndex + j)*dimensionalitypad]);
            // }
            // if (NShared[2 + j*8] != N[k + 2 + (tileIndex + j)*dimensionalitypad]){
            //    printf("i, k, Nshared, N 2: %d, %d, %lf, %lf \n", i, k, NShared[2*tileWidth + j], N[k + 2 + (tileIndex + j)*dimensionalitypad]);
            // }
            // if (NShared[3 + j*8] != N[k + 3 + (tileIndex + j)*dimensionalitypad]){
            //    printf("i, k, Nshared, N 3: %d, %d, %lf, %lf \n", i, k, NShared[3*tileWidth + j], N[k + 3 + (tileIndex + j)*dimensionalitypad]);
            // }
            // if (NShared[4 + j*8] != N[k + 4 + (tileIndex + j)*dimensionalitypad]){
            //    printf("i, k, Nshared, N 4: %d, %d, %lf, %lf \n", i, k, NShared[4*tileWidth + j], N[k + 4 + (tileIndex + j)*dimensionalitypad]);
            // }
            // if (NShared[5 + j*8] != N[k + 5 + (tileIndex + j)*dimensionalitypad]){
            //    printf("i, k, Nshared, N 5: %d, %d, %lf, %lf \n", i, k, NShared[5*tileWidth + j], N[k + 5 + (tileIndex + j)*dimensionalitypad]);
            // }
            // if (NShared[6 + j*8] != N[k + 6 + (tileIndex + j)*dimensionalitypad]){
            //    printf("i, k, Nshared, N 6: %d, %d, %lf, %lf \n", i, k, NShared[6*tileWidth + j], N[k + 6 + (tileIndex + j)*dimensionalitypad]);
            // }
            // if (NShared[7 + j*8] != N[k + 7 + (tileIndex + j)*dimensionalitypad]){
            //    printf("i, k, Nshared, N, NT 7: %d, %d, %lf, %lf, %lf \n", i, k, NShared[7*tileWidth + j], N[k + 7 + (tileIndex + j)*dimensionalitypad], NT[(tileIndex + j) + (k + 7)*n]);
            // }
            //if (i != j){
            force0 += (staticNPos0 - NShared[0 + j*8])*partials[i*n + tileIndex + j];
            force1 += (staticNPos1 - NShared[1 + j*8])*partials[i*n + tileIndex + j];
            force2 += (staticNPos2 - NShared[2 + j*8])*partials[i*n + tileIndex + j];
            force3 += (staticNPos3 - NShared[3 + j*8])*partials[i*n + tileIndex + j];
            force4 += (staticNPos4 - NShared[4 + j*8])*partials[i*n + tileIndex + j];
            force5 += (staticNPos5 - NShared[5 + j*8])*partials[i*n + tileIndex + j];
            force6 += (staticNPos6 - NShared[6 + j*8])*partials[i*n + tileIndex + j];
            force7 += (staticNPos7 - NShared[7 + j*8])*partials[i*n + tileIndex + j];
            //}
            // force0 += (staticNPos0 - NShared[0*32 + j])*partials[i*n + tileIndex + j];
            // force1 += (staticNPos1 - NShared[1*32 + j])*partials[i*n + tileIndex + j];
            // force2 += (staticNPos2 - NShared[2*32 + j])*partials[i*n + tileIndex + j];
            // force3 += (staticNPos3 - NShared[3*32 + j])*partials[i*n + tileIndex + j];
            // force4 += (staticNPos4 - NShared[4*32 + j])*partials[i*n + tileIndex + j];
            // force5 += (staticNPos5 - NShared[5*32 + j])*partials[i*n + tileIndex + j];
            // force6 += (staticNPos6 - NShared[6*32 + j])*partials[i*n + tileIndex + j];
            // force7 += (staticNPos7 - NShared[7*32 + j])*partials[i*n + tileIndex + j];
            // force0 += (staticNPos0 - N[k + 0 + (tileIndex + j)*dimensionality])*partials[i*n + (j + tileIndex)];
            // force1 += (staticNPos1 - N[k + 1 + (tileIndex + j)*dimensionality])*partials[i*n + (j + tileIndex)];
            // force2 += (staticNPos2 - N[k + 2 + (tileIndex + j)*dimensionality])*partials[i*n + (j + tileIndex)];
            // force3 += (staticNPos3 - N[k + 3 + (tileIndex + j)*dimensionality])*partials[i*n + (j + tileIndex)];
            // force4 += (staticNPos4 - N[k + 4 + (tileIndex + j)*dimensionality])*partials[i*n + (j + tileIndex)];
            // force5 += (staticNPos5 - N[k + 5 + (tileIndex + j)*dimensionality])*partials[i*n + (j + tileIndex)];
            // force6 += (staticNPos6 - N[k + 6 + (tileIndex + j)*dimensionality])*partials[i*n + (j + tileIndex)];
            // force7 += (staticNPos7 - N[k + 7 + (tileIndex + j)*dimensionality])*partials[i*n + (j + tileIndex)];
         }
      }
      //printf("tileWidth: %d \n", tileWidth);
      // printf("i: k: force7: %d, %d, %d \n", i, k, force7);
      if (i < n){
         if (k+0 < dimensionality){
            V[i*dimensionality + k + 0] += force0;
         }
         if (k+1 < dimensionality){
            V[i*dimensionality + k + 1] += force1;
         }
         if (k+2 < dimensionality){
            V[i*dimensionality + k + 2] += force2;
         }
         if (k+3 < dimensionality){
            V[i*dimensionality + k + 3] += force3;
         }
         if (k+4 < dimensionality){
            V[i*dimensionality + k + 4] += force4;
         }
         if (k+5 < dimensionality){
            V[i*dimensionality + k + 5] += force5;
         }
         if (k+6 < dimensionality){
            V[i*dimensionality + k + 6] += force6;
         }
         if (k+7 < dimensionality){
            V[i*dimensionality + k + 7] += force7;
         }
      }
   }


   // if (i == 0){
   //    for (int ii = 0; ii < n; ii++){
   //       for (int jj = 0; jj < dimensionality; jj++){
   //          printf("%lf ", V[ii*dimensionality + jj]);
   //       }
   //       printf("\n");
   //    }
   // }
}

__global__ void setup_kernel(curandState *state, int n, int d){
   int id = threadIdx.x + blockIdx.x * 1024;
   if (id < n*d){
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(12345, id, 0, &state[id]);
   }
}

__global__ void generate_normal_kernel(curandState *state, int n, int d, double *result) {
    int id = threadIdx.x + blockIdx.x * 1024;
    if (id < n*d){
       unsigned int count = 0;
       double x;
       /* Copy state to local memory for efficiency */
       curandState localState = state[id];
       /* Generate pseudo-random normals */
       //for(int i = 0; i < n/2; i++) {
           x = curand_uniform(&localState);
   //         /* Check if within one standard deviaton */
   //         if((x.x > -1.0) && (x.x < 1.0)) {
   //             count++;
   //         }
   //         if((x.y > -1.0) && (x.y < 1.0)) {
   //             count++;
   //         }
   //     }
       /* Copy state back to global memory */
       state[id] = localState;
       /* Store results */
       //result[id] += count;
       result[id] = x;
    }
}

__global__ void addVel(double* N, double* V, double momentumConstant, double timeStepConstant, unsigned dimReduceCutoff, unsigned dimensionality, unsigned n, double* randoms){
    int j = threadIdx.x + blockIdx.y * 1024;
    int i = blockIdx.x;
    //printf("i, j: %d, %d\n", i, j);
    if ((j < dimensionality) && (i < n)){
       double distToMove = V[j + i*dimensionality];
       //printf("force, rand: %lf, %lf\n", distToMove, randoms[j + i*dimensionality]);
       //printf("V, i, j: %lf, %d, %d\n", V[j + i*dimensionality], i, j);
       //V[j + i*dimensionality] = distToMove + randoms[j + i*dimensionality];
       V[j + i*dimensionality] = distToMove * momentumConstant;
       N[j + i*dimensionality] += distToMove * timeStepConstant;
       if (j >= dimReduceCutoff){
          N[j + i*dimensionality] *= 0.999;
       }
    }
}

__global__ void addDisplacement(double* distSum, double* V, double timeStepConstant, unsigned dimensionality, unsigned n){
    int i = threadIdx.x + blockIdx.x * 1024;
    //printf("n, d: %u, %u\n", n, dimensionality);
    if (i < n){
       double distToMove = 0.0;
       //printf("%d\n", i);
       for (int j = 0; j < dimensionality; j++){
          //printf("i,j: %d, %d\n", i, j);
          //printf("addr: %d\n", j + i*dimensionality);
          double value = V[j + i*dimensionality];
          //printf("value: %lf\n", value);
          distToMove += value*value;
          //printf("distToMoveIN: %lf\n", distToMove);
          //distToMove += abs(V[j + i*dimensionality]);
       }
       distToMove = sqrt(distToMove);
       //printf("distToMove: %lf\n", distToMove);
       //printf("af%d\n", i+n);
       distSum[i] += distToMove;
    }
}

__global__
void getErrorEnergy(double* error, double* energy, double* W, double* currentDists, unsigned n){
   //printf("%d %d %d %d\n", threadIdx.x, blockIdx.x, threadIdx.y, blockIdx.y);
   int i = threadIdx.x + blockIdx.x * 32;
   int j = threadIdx.y + blockIdx.y * 32;
   if ((i < n) && (j < n) && (j > i)){
      double currentDist = currentDists[j*n + i];
      atomicAdd(energy, currentDist);
      //*energy += 1 / (currentDist*currentDist);
      if (!(W[j*n + i] != W[j*n + i])){
         atomicAdd(error, fabs(currentDist - W[j*n + i]));
         //*error += fabs(currentDist - W[j*n + i]);
      }
   }
}

__global__
//void getMaxEdgeError(double* maxEdgeErrorN, double* W, double* currentDists, unsigned dimensionality, unsigned n){
void getMaxEdgeError(double* maxEdgeErrorN, double* W, double* currentDists, unsigned dimensionality, unsigned n, double* N){
   int i = threadIdx.x + blockIdx.x * 32;
   double maxEdgeError = 0;
   if (i < n){
      for (int j = 0; j < n; j++){
         if (!(W[j*n + i] != W[j*n + i])){
            if (i != j){
               double dist = fabs(currentDists[j*n + i] - W[j*n + i]);
               //if (i == 1){
               //   printf("edgeDist, i, j: %f, %u, %u\n", dist, i, j);
               //}
               if (dist > maxEdgeError){
                  maxEdgeError = dist;
                  // if (dist > 0.25){
                  //    printf("i,j,dist,w: %d, %d, %lf, %lf, %lf %lf %lf %lf %lf %lf\n", i, j, currentDists[j*n + i], W[j*n + i],
                  //       N[i*dimensionality + 0], N[i*dimensionality + 1], N[i*dimensionality + 2],
                  //       N[j*dimensionality + 0], N[j*dimensionality + 1], N[j*dimensionality + 2]);
                  // }
               }
            }
         }
      }
      maxEdgeErrorN[i] = maxEdgeError;
   }
}

void matmulCuBLAS(double* N, double* NT, double* W, double* V, double* Ncu, double* NTcu, double* Wcu, double* Vcu, unsigned dimensionalitypad, unsigned d, unsigned n, double c0, double c1, double momentumConstant, double* error, double* energy, double timeStepConstant, double* maxEdgeError, int iterationNum, unsigned dimReduceCutoff, double* NCopyCu, double* NTCopyCu, double* partials, double* N2, double* ik, double* sum, double* devPtrA, double* devPtrB, double* devPtrC, double* C, double* errorCu, double* energyCu, cublasHandle_t handle, cublasStatus_t cublasStat, curandState *devStates, double *devResults, double* distSum, double edgeEase){
// //    cublasHandle_t handle;
// //    cublasStatus_t cublasStat = cublasCreate(&handle);
// //    cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
   cudaError_t err;
// 
//    *error = 90.0;
//    *energy = 20.0;
//    printf("errorin: %f \n", *error);
//    printf("energyin: %f \n", *energy);

   int rowsA = n;
   int colsA = dimensionalitypad;
  
   //size_t inputMatSize = n*d * sizeof(double);
   size_t matrixSizeA = (size_t)dimensionalitypad * n;

   //printf("d: %u \ndimensionalitypad: %d \nn: %u\n", d, dimensionalitypad, n);
   
   // double* NPrint;
   // cudaMallocHost((double **) &NPrint, n*dimensionalitypad*sizeof(double));
   // cudaMemcpy(NPrint, Ncu, n*d * sizeof(double), cudaMemcpyDeviceToHost);
   // printf("N : \n");
   // for (int i = 0; i < n; i++){
   //    for (int j = 0; j < d; j++){
   //       printf("%lf ", NPrint[i*d + j]);
   //    }
   //    printf("\n");
   // }
   // cudaFreeHost(NPrint);
   // double* NTPrint;
   // cudaMallocHost((double **) &NTPrint, n*d*sizeof(double));
   // cudaMemcpy(NTPrint, NTcu, n*d * sizeof(double), cudaMemcpyDeviceToHost);
   // printf("NT : \n");
   // for (int i = 0; i < d; i++){
   //    for (int j = 0; j < n; j++){
   //       printf("%lf ", NTPrint[i*n + j]);
   //    }
   //    printf("\n");
   // }
   // cudaFreeHost(NTPrint);
   // fflush(stdout);



   ////double* NCopy = NULL;
   ////double* NTCopy = NULL;
   ////double* NCopyCu = NULL;
   ////double* NTCopyCu = NULL;
   double zero = 0.0;
   //double* zeroPtr = &zero;
   ////cudaMallocHost((void**) &NCopy, matrixSizeA * sizeof(double));
   ////cudaMallocHost((void**) &NTCopy, matrixSizeA * sizeof(double));
   ////cudaMalloc((void**) &NCopyCu, matrixSizeA * sizeof(double));
   ////cudaMalloc((void**) &NTCopyCu, matrixSizeA * sizeof(double));
   unsigned blockNum = n / 32;
   if (n % 32 != 0){
      blockNum++;
   }
   dim3 blocks = dim3(blockNum, blockNum);
   dim3 threads = dim3(32, 32);
   copyToPadded<<<blocks, threads>>>(NCopyCu, Ncu, dimensionalitypad, d, n);
   cudaDeviceSynchronize();
   err = cudaGetLastError();  // add
   if (err != cudaSuccess){
      std::cout << "CUDA error: " << cudaGetErrorString(err) << " pad N " << std::endl; // add
      return;
   }
   copyToPaddedT<<<blocks, threads>>>(NTCopyCu, NTcu, dimensionalitypad, d, n);
   cudaDeviceSynchronize();
   err = cudaGetLastError();  // add
   if (err != cudaSuccess){
      std::cout << "CUDA error: " << cudaGetErrorString(err) << " pad NT " << std::endl; // add
      return;
   }
// //    cudaMemcpy(NCopy, NCopyCu, matrixSizeA * sizeof(double), cudaMemcpyDeviceToHost);
// //    err = cudaGetLastError();  // add
// //    if (err != cudaSuccess){
// //       std::cout << "CUDA error: " << cudaGetErrorString(err) << " Ncopy memcpy " << std::endl; // add
// //       return;
// //    }
// //    cudaMemcpy(NTCopy, NTCopyCu, matrixSizeA * sizeof(double), cudaMemcpyDeviceToHost);
// //    err = cudaGetLastError();  // add
// //    if (err != cudaSuccess){
// //       std::cout << "CUDA error: " << cudaGetErrorString(err) << " NTcopy memcpy " << std::endl; // add
// //       return;
// //    }

// //    double *devPtrA = NULL;
// //    cudaMalloc((void**) &devPtrA, matrixSizeA * sizeof(double));
// //    err = cudaGetLastError();  // add
// //    if (err != cudaSuccess){
// //       std::cout << "CUDA error: " << cudaGetErrorString(err) << " a alloc " << std::endl; // add
// //       //cuProfilerStop();
// //       return;
// //    }
   //double* A  = (double*) malloc(matrixSizeA * sizeof(double));
// //    double* A = NULL;
// //    cudaMalloc((void**) &A, matrixSizeA * sizeof(double));
// //    if (!A) {
// //         printf ("host memory allocation failed");
// //         return;
// //    }
// //    //memset(A, 0x0000, matrixSizeA* sizeof(double));
// //    cudaMemcpy(A, NCopy, matrixSizeA * sizeof(double), cudaMemcpyHostToDevice);
// // //    cudaMemcpy(A, NCopyCu, matrixSizeA * sizeof(double), cudaMemcpyDeviceToDevice);
// //    cublasStatus_t statusA = cublasSetMatrix(rowsA, colsA, sizeof(double), A, rowsA, devPtrA, rowsA);
   cublasStatus_t statusA = cublasSetMatrix(rowsA, colsA, sizeof(double), NCopyCu, rowsA, devPtrA, rowsA);

   err = cudaGetLastError();  // add
   if (err != cudaSuccess){
      std::cout << "CUDA error: " << cudaGetErrorString(err) << " a copy " << std::endl; // add
      //cuProfilerStop();
      return;
   }


// //    double *devPtrB = NULL;
// //    cudaMalloc((void**)&devPtrB, matrixSizeA * sizeof(double));
// //    double* B  = NULL;
// //    cudaMalloc((void**) &B, matrixSizeA * sizeof(double));
// // // //    //memset( B, 0x0000, matrixSizeA* sizeof(double));
// //    cudaMemcpy(B, NTCopy, matrixSizeA * sizeof(double), cudaMemcpyHostToDevice);
// // // //    cudaMemcpy(B, NTCopyCu, matrixSizeA * sizeof(double), cudaMemcpyDeviceToDevice);
   cublasStatus_t statusB = cublasSetMatrix(colsA, rowsA, sizeof(double), NTCopyCu, colsA, devPtrB, colsA);
// //    cublasStatus_t statusB = cublasSetMatrix(colsA, rowsA, sizeof(double), B, colsA, devPtrB, colsA);

   size_t matrixSizeC = n * n;
// //    double *devPtrC = 0;
// //    cudaMalloc((void**)&devPtrC, matrixSizeC * sizeof(double));
// //    //double* C  = (double *)malloc(matrixSizeC * sizeof(double));
// //    double* C = NULL;
// //    cudaMalloc((void**) &C, matrixSizeC * sizeof(double));
   //memset( C, 0x0000, matrixSizeC * sizeof(double));
   cudaMemset(C, zero, matrixSizeC * sizeof(double));
   cublasStatus_t statusC = cublasSetMatrix(rowsA, rowsA, sizeof(double), C, rowsA, devPtrC, rowsA);

   if (statusA != cudaSuccess) {
        printf ("device memory allocation failed A\n");
        return;
   }
   if (statusB != cudaSuccess) {
        printf ("device memory allocation failed B\n");
        return;
    }
   if (statusC != cudaSuccess) {
        printf ("device memory allocation failed C\n");
        return;
   }

   cublasOperation_t transa =  CUBLAS_OP_N;
   cublasOperation_t transb =  CUBLAS_OP_N;
   cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
   double alpha = -2.0;//const void* alpha;//int alpha = -2;
   double beta = 1.0;//const void* beta;//int beta = 1;
// //    cublasStat = cublasGemmEx(handle, transa, transb, n, n, dimensionalitypad, &alpha,
// //                           B, CUDA_R_32F, dimensionalitypad,
// //                           A, CUDA_R_32F, n,
// //                           &beta, C, CUDA_R_32F, n, CUDA_R_32F, algo);
   //printf("n, dimpad: %u, %u\n", n, dimensionalitypad);
   cublasStat = cublasGemmEx(handle, transa, transb, n, n, dimensionalitypad, &alpha,
                          NTCopyCu, CUDA_R_64F, n,
                          NCopyCu, CUDA_R_64F, dimensionalitypad,
                          &beta, C, CUDA_R_64F, n, CUDA_R_64F, algo);

   cudaDeviceSynchronize();

// //    double* N2 = NULL;
// //    double* ik = NULL;
// //    double* sum = NULL;
// //    cudaMalloc((void**) &N2, matrixSizeA * sizeof(double));
// //    cudaMalloc((void**) &ik, n * sizeof(double));
// //    cudaMalloc((void**) &sum, n*n*sizeof(double));
// //    ptwSquare<<<(n*n)/1024 + 1, 1024>>>(N2, A, n, dimensionalitypad);
   ptwSquare<<<(n*n)/1024 + 1, 1024>>>(N2, NCopyCu, n, dimensionalitypad);
   cudaDeviceSynchronize();
   err = cudaGetLastError();  // add
   if (err != cudaSuccess){
      std::cout << "CUDA error: " << cudaGetErrorString(err) << " square " << std::endl; // add
      //cuProfilerStop();
      return;
   }
   blockNum = n/1024;
   if (n % 1024 != 0){
      blockNum++;
   }
   //printf("blockNum: %u \n", blockNum);
   squareReduce<<<blockNum, 1024>>>(ik, N2, n, dimensionalitypad);
   cudaDeviceSynchronize();
   err = cudaGetLastError();  // add
   if (err != cudaSuccess){
      std::cout << "CUDA error: " << cudaGetErrorString(err) << " reduce " << std::endl; // add
      //cuProfilerStop();
      return;
   }
   blockNum = n / 32;
   if (n % 32 != 0){
      blockNum++;
   }
   blocks = dim3(blockNum, blockNum);
   threads = dim3(32, 32);
   distSums<<<blocks, threads>>>(sum, ik, C, n);
   cudaDeviceSynchronize();
   err = cudaGetLastError();  // add
   if (err != cudaSuccess){
      std::cout << "CUDA error: " << cudaGetErrorString(err) << " sum " << std::endl; // add
      //cuProfilerStop();
      return;
   }

   ////double* partials = NULL;
   ////cudaMalloc((void**) &partials, n*n*sizeof(double));
   partialComp<<<blocks, threads>>>(partials, sum, c0, n);
   cudaDeviceSynchronize();
   err = cudaGetLastError();  // add
   if (err != cudaSuccess){
      std::cout << "CUDA error: " << cudaGetErrorString(err) << " partials " << std::endl; // add
      return;
   }

   
   dim3 dimBlockSize(n, (d + 1024-1)/1024);
   int dimN = n / 512;
   int dimD = d / 8;
   if (n % 512 != 0){
      dimN++;
   }
   if (d % 8 != 0){
      dimD++;
   }
   dim3 tileDims(dimN,dimD);
   NBodyMul<<<tileDims, 512>>>(NCopyCu, NTCopyCu, Vcu, partials, d, dimensionalitypad, n);
   //NBodyMul<<<dimBlockSize, 1024, n*sizeof(double)>>>(Ncu, NTcu, Vcu, partials, d, n);
   //printf("n: %d\n", n);
   //NBodyMul<<<dimBlockSize, 1024, n*32*sizeof(double)>>>(Ncu, NTcu, Vcu, partials, d, n);
   cudaDeviceSynchronize();
   err = cudaGetLastError();  // add
   if (err != cudaSuccess){
       std::cout << "CUDA error: " << cudaGetErrorString(err) << " nbodymul " << std::endl; // add
       return;
   }

   edgeForces<<<dimBlockSize, 1024>>>(Wcu, Ncu, NTcu, Vcu, sum, d, n, c1, edgeEase);
   cudaDeviceSynchronize();
     err = cudaGetLastError();  // add
     if (err != cudaSuccess){
         std::cout << "CUDA error: " << cudaGetErrorString(err) << " eF " << std::endl; // add
         return;
     }

   // random things
   generate_normal_kernel<<<(n*n)/1024 + 1, 1024>>>(devStates, n, dimensionalitypad, devResults);
//           __device__ void
/*curand_init (
    unsigned long long seed, unsigned long long sequence,
    unsigned long long offset, curandState_t *state)*/
    
           

  addDisplacement<<< n/1024 + 1, 1024>>>(distSum, Vcu, timeStepConstant, d, n);


   //double momentumTimeConstant = momentumConstant * timeStepConstant;
   addVel<<<dimBlockSize, 1024>>>(Ncu, Vcu, momentumConstant, timeStepConstant, dimReduceCutoff, d, n, devResults);
   cudaDeviceSynchronize();
     err = cudaGetLastError();  // add
     if (err != cudaSuccess){
         std::cout << "CUDA error: " << cudaGetErrorString(err) << " VelAdd " << std::endl; // add
         return;
     }


 

// //    double* errorCu = NULL;
// //    double* energyCu = NULL;
// //    cudaMalloc((void**) &errorCu, sizeof(double));
// //    cudaMalloc((void**) &energyCu, sizeof(double));
   cudaDeviceSynchronize();
   err = cudaGetLastError();  // add
     if (err != cudaSuccess){
         std::cout << "CUDA error: " << cudaGetErrorString(err) << " e alloc " << std::endl; // add
         return;
     }
   //int zeroInt = 0;
   cudaMemset(errorCu, zero, sizeof(double));
   cudaMemset(energyCu, zero, sizeof(double));
   cudaDeviceSynchronize();
     err = cudaGetLastError();  // add
     if (err != cudaSuccess){
         std::cout << "CUDA error: " << cudaGetErrorString(err) << " memeset " << std::endl; // add
         return;
     }
   getErrorEnergy<<<blocks, threads>>>(errorCu, energyCu, Wcu, sum, n);
   cudaDeviceSynchronize();
     err = cudaGetLastError();  // add
     if (err != cudaSuccess){
         std::cout << "CUDA error: " << cudaGetErrorString(err) << " error/energy " << std::endl; // add
         return;
     }
   cudaMemcpy(error, errorCu, sizeof(double), cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();
     err = cudaGetLastError();  // add
     if (err != cudaSuccess){
         std::cout << "CUDA error: " << cudaGetErrorString(err) << " error copy " << std::endl; // add
         return;
     }
   cudaMemcpy(energy, energyCu, sizeof(double), cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();
     err = cudaGetLastError();  // add
     if (err != cudaSuccess){
         std::cout << "CUDA error: " << cudaGetErrorString(err) << " VelAdd or error/energy " << std::endl; // add
         return;
     }


   if ((iterationNum % 1000) == 0){
      double* maxEdgeErrorNCu = NULL;
      double* maxEdgeErrorN = NULL;
      cudaMalloc((void**) &maxEdgeErrorNCu, n*sizeof(double));
      cudaMallocHost((void**) &maxEdgeErrorN, n*sizeof(double));
      getMaxEdgeError<<<blocks, threads>>>(maxEdgeErrorNCu, W, sum, d, n, Ncu);
      cudaDeviceSynchronize();
      cudaMemcpy(maxEdgeErrorN, maxEdgeErrorNCu, n*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      double maxEdgeErrorLocal = 0.0;
      for (int i = 0; i < n; i++){
         //printf("mEE, i: %d, %f\n", i, maxEdgeError);
         if (maxEdgeErrorN[i] > maxEdgeErrorLocal){
            maxEdgeErrorLocal = maxEdgeErrorN[i];
         }
      }
      *maxEdgeError = maxEdgeErrorLocal;
      err = cudaGetLastError();  // add
        if (err != cudaSuccess){
            std::cout << "CUDA error: " << cudaGetErrorString(err) << " maxEdgeError " << std::endl; // add
            return;
        }
      printf("%u\n", iterationNum);
      printf("error: %lf, maxee: %lf\n", *error, *maxEdgeError);
      printf("energy: %lf\n", *energy);
      fflush(stdout);
      cudaFree(maxEdgeErrorNCu);
      cudaFreeHost(maxEdgeErrorN);
   }


   // printf("NCopy : \n");
   // for (int i = 0; i < n; i++){
   //    for (int j = 0; j < dimensionalitypad; j++){
   //       //printf("%f ", N[i*d + j]);
   //       //printf("%f ", A[i*d + j]);
   //       printf("%f ", NCopy[i*dimensionalitypad + j]);
   //    }
   //    printf("\n");
   // }
//    printf("NTCopy : \n");
//    for (int i = 0; i < dimensionalitypad; i++){
//       for (int j = 0; j < n; j++){
//          //printf("%f ", NT[i*n + j]);
//          printf("%f ", NTCopy[i*n + j]);
//       }
//       printf("\n");
//    }
//            double* TEST = NULL;
//    cudaMallocHost((void**) &TEST, matrixSizeA * sizeof(double));
//    cudaMemcpy(TEST, A, matrixSizeA * sizeof(double), cudaMemcpyDeviceToHost);
//    printf("A : \n");
//    for (int i = 999; i < n; i++){
//       for (int j = 0; j < dimensionalitypad; j++){
//          //printf("%f ", N[i*d + j]);
//          //printf("%f ", A[i*d + j]);
//          printf("%f ", TEST[i*dimensionalitypad + j]);
//       }
//       printf("\n");
//    }
//    double* TESTB = NULL;
//    cudaMallocHost((void**) &TESTB, matrixSizeA * sizeof(double));
//    cudaMemcpy(TESTB, B, matrixSizeA * sizeof(double), cudaMemcpyDeviceToHost);
//    printf("B : \n");
//    for (int i = 0; i < dimensionalitypad; i++){
//       for (int j = 0; j < n; j++){
//          //printf("%f ", NT[i*n + j]);
//          printf("%f ", TESTB[i*n + j]);
//       }
//       printf("\n");
//    }
   // double* TESTC = NULL;
   // cudaMallocHost((void**) &TESTC, n*n*sizeof(double));
   // cudaMemcpy(TESTC, C, n*n * sizeof(double), cudaMemcpyDeviceToHost);
   // printf("C : \n");
   // for (int i = 0; i < n; i++){
   //    for (int j = 0; j < n; j++){
   //       //printf("%f ", NT[i*n + j]);
   //       printf("%f ", TESTC[i*n + j]);
   //    }
   //    printf("\n");
   // }
//    double* TESTN2 = NULL;
//    cudaMallocHost((void**) &TESTN2, matrixSizeA*sizeof(double));
//    cudaMemcpy(TESTN2, N2, matrixSizeA * sizeof(double), cudaMemcpyDeviceToHost);
//    printf("N2 : \n");
//    for (int i = 999; i < dimensionalitypad; i++){
//       for (int j = 0; j < n; j++){
//          //printf("%f ", NT[i*n + j]);
//          printf("%f ", TESTN2[i*dimensionalitypad + j]);
//       }
//       printf("\n");
//    }
//    double* TESTik = NULL;
//    cudaMallocHost((void**) &TESTik, n*sizeof(double));
//    cudaMemcpy(TESTik, ik, n * sizeof(double), cudaMemcpyDeviceToHost);
//    printf("ik : \n");
//    for (int i = 0; i < n; i++){
//       printf("%f ", TESTik[i]);
//       printf("\n");
//    }
   // double* currentDists;
   // cudaMallocHost((double **) &currentDists, n*n*sizeof(double));
   // cudaMemcpy(currentDists, sum, n*n * sizeof(double), cudaMemcpyDeviceToHost);
   // printf("currentDists : \n");
   // for (int i = 0; i < n; i++){
   //    for (int j = 0; j < n; j++){
   //       printf("%lf ", currentDists[i*n + j]);
   //    }
   //    printf("\n");
   // }
   // cudaFreeHost(currentDists);
   // double* NPrint;
   // cudaMallocHost((double **) &NPrint, n*dimensionalitypad*sizeof(double));
   // cudaMemcpy(NPrint, Ncu, n*d * sizeof(double), cudaMemcpyDeviceToHost);
   // printf("N : \n");
   // for (int i = 0; i < n; i++){
   //    for (int j = 0; j < d; j++){
   //       printf("%lf ", NPrint[i*d + j]);
   //    }
   //    printf("\n");
   // }
   // cudaFreeHost(NPrint);
   // double* NTPrint;
   // cudaMallocHost((double **) &NTPrint, n*d*sizeof(double));
   // cudaMemcpy(NTPrint, NTcu, n*d * sizeof(double), cudaMemcpyDeviceToHost);
   // printf("NT : \n");
   // for (int i = 0; i < d; i++){
   //    for (int j = 0; j < n; j++){
   //       printf("%lf ", NTPrint[i*n + j]);
   //    }
   //    printf("\n");
   // }
   // cudaFreeHost(NTPrint);



// //    cudaFree(energyCu);
// //    cudaFree(errorCu);
   ////cudaFree(partials);
// //    cudaFree(N2);
// //    cudaFree(ik);
// //    cudaFree(sum);
// //    cudaFree(devPtrB);
// //    cudaFree(B);
// //    cudaFree(devPtrC);
// //    cudaFree(C);
// //    cudaFree(devPtrA);
// //    cudaFree(A);
   ////cudaFreeHost(NCopy);
   ////cudaFreeHost(NTCopy);
   ////cudaFree(NCopyCu);
   ////cudaFree(NTCopyCu);
   ////cublasDestroy(handle);
   cudaDeviceSynchronize();
     err = cudaGetLastError();  // add
     if (err != cudaSuccess){
         std::cout << "CUDA error: " << cudaGetErrorString(err) << " frees " << std::endl; // add
         return;
     }
}

double optimizeC(double* N, double* W, unsigned dimensionality, unsigned n, double* errors, double* energies, double* maxEdgeErrors, unsigned dimReduceCutoff, double* distSum){
    printf("dim, dimReduce: %u, %u\n", dimensionality, dimReduceCutoff);
    printf("n: %u\n", n);
    bool errorBool = 0;
    double c0 = 0.005;//0.4/double(n); // 400 node 0.0005 //200 nodes 0.001 //0.000008  // 0.0001
    printf("c0: %lf\n", c0);
    double c1 = 1; // 0.01 // 0.5
    printf("c1: %lf\n", c1);
    double momentumConstant = 0.9; //0.8 // 0.999
    double timeStepConstant = 0.05;//0.001/double(n); // 0.001 0.00001
    // hueristic based stoping
    // unsigned numIterations = 150000; //500 //20000
    // if (dimReduceCutoff < dimensionality){
    //    numIterations = 5000;
    // }
    unsigned iterationsBeforeJump = n;//n/10; // 100
    unsigned itCurrentCount = 0;
    // unsigned itsBeforeFirstJump = 120000;
    // if (dimReduceCutoff < dimensionality){
    //    itsBeforeFirstJump = 4000;
    //    iterationsBeforeJump = 200;
    // }
    // printf("numIterations: %u\n", numIterations);
    double learningRate = 1.001; // .005
    double maxEdgeErrorStopPoint = 0.005;
    unsigned itsSlowForAnnealing = 5000;
    unsigned slowEnergyIts = 0;
    double energyRate = 1.00001;
    unsigned numIterations = 40000; // upperbound of iterations
    bool annealing = 0;
    if (dimReduceCutoff == dimensionality){
       numIterations = 40000;
       itsSlowForAnnealing = 5000;
    }

    double edgeEase = 1;//0.5;
    int easeCount = 0;
    int easeIts = 300;
    //double learningAcc = 0.99;
    //double errorVel = 0;
    //double prevErrorVel = 1.01;
    printf("learningRate: %lf\n", learningRate);
    //double V[dimensionality][n] = {0};
    double* V = (double*)calloc(dimensionality*n, sizeof(double));


    double* error = NULL;
    double* energy = NULL;
    double* maxEdgeError = NULL;
    cudaMallocHost((void **) &error, sizeof(double));
    cudaMallocHost((void **) &energy, sizeof(double));
    cudaMallocHost((void **) &maxEdgeError, sizeof(double));
    double prevError = FLT_MAX;
    double prevEnergy = DBL_MAX;
    unsigned t = 0;
    //unsigned i = 0;
    //unsigned j = 0;
    //unsigned k = 0;
    double* Ncu;
    double* Vcu;
    double* Wcu;
    double* currentDistscu;
    double* NTcu;
    size_t matSize = n*dimensionality*sizeof(double);
    size_t pairsMatSize = n*n*sizeof(double);
    printf("matSize: %zu\n", matSize);
    cudaMalloc((void**)&currentDistscu, pairsMatSize);
    cudaMalloc((void**)&NTcu, matSize);
    double* NT = NULL;//(double*) malloc(matSize);
    double* currentDists = NULL;//(double*) malloc(pairsMatSize);
    cudaMallocHost((void**)&NT, matSize);
    cudaMallocHost((void**)&currentDists, pairsMatSize);
    cudaMalloc((void**)&Ncu, matSize);
    cudaMalloc((void**)&Vcu, matSize);
    cudaMalloc((void**)&Wcu, pairsMatSize);
    cudaMemcpy(Wcu, W, pairsMatSize, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess){
        std::cout << "CUDA error: " << cudaGetErrorString(err) << " memcpy1 " << std::endl; // add
        return errorBool;
    }
    cudaMemcpy(Ncu, N, matSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Vcu, V, matSize, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
        err = cudaGetLastError();  // add
        if (err != cudaSuccess){
            std::cout << "CUDA error: " << cudaGetErrorString(err) << " memcpy2 " << std::endl; // add
            //cuProfilerStop();
            return errorBool;
        }

   unsigned dimensionalitypad = dimensionality;
   if (dimensionality % 8 != 0){
     dimensionalitypad += 8 - (dimensionality % 8);
   }
   double* NCopyCu = NULL;
   double* NTCopyCu = NULL;
   cudaMalloc((void**) &NCopyCu, n*dimensionalitypad * sizeof(double));
   cudaMalloc((void**) &NTCopyCu, n*dimensionalitypad * sizeof(double));

   double* partials = NULL;
   cudaMalloc((void**) &partials, n*n*sizeof(double));
   double* N2 = NULL;
   double* ik = NULL;
   double* sum = NULL;
   cudaMalloc((void**) &N2, n*dimensionalitypad * sizeof(double));
   cudaMalloc((void**) &ik, n * sizeof(double));
   cudaMalloc((void**) &sum, n*n*sizeof(double));


   double *devPtrA = NULL;
   cudaMalloc((void**) &devPtrA, n*dimensionalitypad * sizeof(double));
   err = cudaGetLastError();  // add
   if (err != cudaSuccess){
      std::cout << "CUDA error: " << cudaGetErrorString(err) << " a alloc " << std::endl; // add
      //cuProfilerStop();
      return errorBool;
   }
   double *devPtrB = NULL;
   cudaMalloc((void**)&devPtrB, n*dimensionalitypad * sizeof(double));
   size_t matrixSizeC = n * n;
   double *devPtrC = 0;
   cudaMalloc((void**)&devPtrC, matrixSizeC * sizeof(double));
   //double* C  = (double *)malloc(matrixSizeC * sizeof(double));
   double* C = NULL;
   cudaMalloc((void**) &C, matrixSizeC * sizeof(double));
   double* errorCu = NULL;
   double* energyCu = NULL;
   cudaMalloc((void**) &errorCu, sizeof(double));
   cudaMalloc((void**) &energyCu, sizeof(double));
   cublasHandle_t handle;
   cublasStatus_t cublasStat = cublasCreate(&handle);
   cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);


   // random things setup
   curandState *devStates;
   double *devResults;
   cudaMalloc((void **)&devResults, matrixSizeC * sizeof(double));
   cudaMemset(devResults, 0, matrixSizeC * sizeof(double));
   cudaMalloc((void **)&devStates, matrixSizeC * sizeof(curandState));          
   setup_kernel<<<(n*n)/1024 + 1, 1024>>>(devStates, n, dimensionalitypad);

   for (t = 1; t <= numIterations; t++){
        //printf("%u\n", t);
//         cudaMemcpy(Ncu, N, matSize, cudaMemcpyHostToDevice);
//         cudaDeviceSynchronize();
//         err = cudaGetLastError();  // add
//         if (err != cudaSuccess){
//             std::cout << "CUDA error: " << cudaGetErrorString(err) << " memcpy2 " << std::endl; // add
//             //cuProfilerStop();
//             return errorBool;
//         }
//         cudaMemcpy(Vcu, V, matSize, cudaMemcpyHostToDevice);
//         err = cudaGetLastError();  // add
//         if (err != cudaSuccess){
//             std::cout << "CUDA error: " << cudaGetErrorString(err) << " memcpy3 " << std::endl; // add
//             return errorBool;
//         }
        
        //printf("gpu\n");
        //printf("dimdiiv: %d \n", int(ceil(dimensionality/1024)));
        //printf("ndiv: %d \n", int(ceil(n/1024)));
        //printf("dimdiiv: %d \n", (dimensionality + 1024-1)/1024);
        //printf("ndiv: %d \n", (n + 1024-1)/1024);
        dim3 dimBlockSize(n, (dimensionality + 1024-1)/1024);
        dim3 nBlockSize(n, (n + 1024-1)/1024);
        transpose<<<dimBlockSize, 1024>>>(Ncu, NTcu, dimensionality, n);
        cudaDeviceSynchronize();
        err = cudaGetLastError();  // add
        if (err != cudaSuccess){
            std::cout << "CUDA error: " << cudaGetErrorString(err) << " tran " << std::endl; // add
            return errorBool;
        }


        //cudaMemcpy(NT, NTcu, matSize, cudaMemcpyDeviceToHost);
        matmulCuBLAS(N, NT, W, V, Ncu, NTcu, Wcu, Vcu, dimensionalitypad, dimensionality, n, c0, c1, momentumConstant, error, energy, timeStepConstant, maxEdgeError, t, dimReduceCutoff, NCopyCu, NTCopyCu, partials, N2, ik, sum, devPtrA, devPtrB, devPtrC, C, errorCu, energyCu, handle, cublasStat, devStates, devResults, distSum, edgeEase);

        
// //         cudaMemcpy(N, Ncu, matSize, cudaMemcpyDeviceToHost);
// //         cudaMemcpy(V, Vcu, matSize, cudaMemcpyDeviceToHost);
// //         cudaMemcpy(currentDists, currentDistscu, pairsMatSize, cudaMemcpyDeviceToHost);
        
        if (annealing){
              //momentumConstant = 0.2;
              //errorVel = (prevError - error) / error;
              //double errorAcc = errorVel / prevErrorVel;
              if (prevError < *error * learningRate){ // && (errorAcc < learningAcc)){
                  itCurrentCount = itCurrentCount + 1;
              }
              if (itCurrentCount > iterationsBeforeJump){
                  itCurrentCount = 0;
                  //momentumConstant *= 0.9;
                  momentumConstant = 0.2;
                  c0 = c0 * 0.78; // 0.7
                  c1 = c1 * 0.8; //0.8

                  // print annealing
                  // printf("c0: %lf\n", c0);
                  // printf("c1: %lf\n", c1);
                  // printf("momentumConstant: %lf\n", momentumConstant);
              }

              if (*maxEdgeError < maxEdgeErrorStopPoint){
                 break;
              }
        } else {
           if ((t % 10000) == 0){
             if (*maxEdgeError > 10){ // 0.01
                timeStepConstant *= 0.9;
             } else {
                timeStepConstant *= 1.1;
             }
           }
           if (prevError * energyRate * c0 < *error * c0){
             slowEnergyIts++;
             if (itsSlowForAnnealing < slowEnergyIts){
                annealing = 1;
             }
           }
        }
        
        // easeCount++;
        // if (easeCount > easeIts){
        //    edgeEase += 0.01;
        //    if (edgeEase > 1.0){
        //        edgeEase = 1.0;
        //    }
        //    easeCount = 0;
        //    printf("edgeEase: %lf\n", edgeEase);
        // }
//         // worst edge fix test
//         if ((t % 100) == 0){
//            fixWorstEdge(N, W, dimensionality, n);
//         }
                
        prevError = *error;
        //prevErrorVel = errorVel;
        prevEnergy = *energy;
        //printf("error: %lf\n", *error);
//         printf("error: %lf, maxee: %lf\n", *error, *maxEdgeError);
//         printf("energy: %lf\n", *energy);

//         printf("Nend:\n");
//         for (int j = 0; j < n; j++){
//         for (int k = 0; k < dimensionality; k++){
//            printf("%f, ", N[j*dimensionality + k]);
//         }
//         printf("\n");
//         }
//         printf("Vend: \n");
//         cudaMemcpy(V, Vcu, matSize, cudaMemcpyDeviceToHost);
//         for (int ii = 0; ii < n; ii++){
//            for (int jj = 0; jj < dimensionality; jj++){
//                printf("%f ", V[ii*dimensionality + jj]);
//            }
//            printf("\n");
//         }
    }

    cudaMemcpy(N, Ncu, matSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(V, Vcu, matSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(currentDists, currentDistscu, pairsMatSize, cudaMemcpyDeviceToHost);


    // check if dimensions being cut under threshold
    bool fit = 1;
    if  (*maxEdgeError > 10000){
      fit = 0;
    } else {
    for (int i = 0; i < n; i++){
       for (int j = dimReduceCutoff; j < dimensionality; j++){
          if (N[j + i*dimensionality] > 9999999999){ // 0.05
             fit = 0;
          }
       }
    }}
    

    errors[dimReduceCutoff] = *error;
    energies[dimReduceCutoff] = *energy;
    maxEdgeErrors[dimReduceCutoff] = *maxEdgeError;
    cudaFree(Ncu);
    cudaFree(Vcu);
    cudaFree(currentDistscu);
    cudaFree(NTcu);
    cudaFree(Wcu);
    cudaFreeHost(NT);
    cudaFreeHost(currentDists);
    cudaFreeHost(error);
    cudaFreeHost(energy);
    cudaFreeHost(maxEdgeError);
    cudaFree(NCopyCu);
    cudaFree(NTCopyCu);
    cudaFree(partials);
    cudaFree(N2);
    cudaFree(ik);
    cudaFree(sum);

   cudaFree(devPtrA);
   cudaFree(devPtrB);
   cudaFree(devPtrC);
   cudaFree(C);
   cudaFree(errorCu);
   cudaFree(energyCu);

   cublasDestroy(handle);

    err = cudaGetLastError();  // add
     if (err != cudaSuccess){
         std::cout << "CUDA error: " << cudaGetErrorString(err) << " free optC " << std::endl; // add
         return errorBool;
     }
     printf("ending error: %lf, maxee: %lf\n", errors[dimReduceCutoff], maxEdgeErrors[dimReduceCutoff]);
    return maxEdgeErrors[dimReduceCutoff];
}


void shiftToCorrectDist(double* N, double* W, unsigned dimensionality, unsigned n){
   double currentDist = 0;
   //double energy = 0;
   int k = 0;
   int j = 0;
   int i = 0;

   printf("shift start N: %p\n", N);

   for (i = 0; i < n; i++){
      for (j = i+1; j < n; j++){//n
        if (i == j){
           k = 0;
        } else {
           // 2 norm
           //double currentDist = 0;
           //unsigned k = 0;
           currentDist = 0;
           for (k = 0; k < dimensionality; k++){
              double dist = N[k + i*dimensionality] - N[k + j*dimensionality];
              currentDist += dist*dist + 0.000001;
           }
           currentDist = sqrt(currentDist);

           if (!(W[i*n + j] != W[i*n + j])){
              for (k = 0; k < dimensionality; k++){
                 double relativeDist = currentDist - W[i*n + j];
                 double force = ((N[k + i*dimensionality] - N[k + j*dimensionality])/currentDist) * relativeDist;
                 if (relativeDist > 0){
                    N[k + i*dimensionality] -= force;
                    N[k + j*dimensionality] += force;
                 } else {
                    N[k + i*dimensionality] += force;
                    N[k + j*dimensionality] -= force; 
                 }
              }
           }
         }
      }
      //printf("N,k,i: %f %d %d\n", N[k + i*dimensionality], k, i);
   }  
   //printf("shift end N: %p\n", N);
}

void setupManifold(double* N, int n, int nActual, int dimensionality, int posDim){
   //FILE* positionFP = fopen("C:\\Users\\Peter Oostema\\Documents\\school\\graphResearch\\Mar312020-20200401T020813Z-001\\Mar312020\\graphEmbed\\graphs\\manifold\\spiral.bin", "rb");
   // FILE* positionFP = fopen("C:\\Users\\Peter Oostema\\Documents\\school\\graphResearch\\Mar312020-20200401T020813Z-001\\Mar312020\\graphEmbed\\graphs\\manifold\\dome.bin", "rb");
   FILE* positionFP = fopen("C:\\Users\\Peter Oostema\\Documents\\school\\graphResearch\\Mar312020-20200401T020813Z-001\\Mar312020\\graphEmbed\\graphs\\manifold\\elipPara.bin", "rb");
   double* positionData = (double*) malloc(sizeof(double)*n*posDim);
   fread(positionData, sizeof(double)*nActual*posDim,1,positionFP);

   double random_value;
   srand(time(NULL));

   for (int i = 0; i < nActual; i++){
      for (int j = 0; j < dimensionality; j++){
         if (j < posDim){
            N[j + i*dimensionality] = positionData[i*posDim + j];
         } else {
            N[j + i*dimensionality] = 0.0;
            //random_value = (double)rand()/RAND_MAX*0.02 - 0.01;
            //N[j + i*dimensionality] = random_value;
         }
      }
   }
   for (int i = nActual; i < n; i++){
      for (int j = 0; j < dimensionality; j++){
         N[j + i*dimensionality] = (double)rand() * 1000000;
      }
   }
   
   free(positionData);
   printf("N\n");
   for (int i = 0; i < n; i++){
      for (int j = 0; j < posDim; j++){
         printf("%lf ", N[j + i*dimensionality]);
      }
      printf("\n");
   }
}

void setupLowDim(double* N, int n, int dimensionality, int dimCutoff){
   double random_value;
   srand(time(NULL));
   for (int j = 0; j < n; j++){
      for (int k = 0; k < dimCutoff; k++){
         random_value = (double)rand()/RAND_MAX*2.0 - 1.0;
         N[k + j*dimensionality] = random_value;
      }
      for (int k = dimCutoff; k < dimensionality; k++){
         N[k + j*dimensionality] = 0.0;
      }
   }
}

void setupSimplex(double* N, int n, int dimensionality){
   double* midpoint = (double*) calloc(dimensionality, sizeof(double));
   for (int i = 1; i < n; i++){
      // find midpoint of points already placed
      for (int j = 0; j < n; j++){
         for (int k = 0; k < dimensionality; k++){
            midpoint[k] += N[k + j*dimensionality];
         }
      }
      for (int k = 0; k < dimensionality; k++){
         midpoint[k] /= i;
      }
      
      double currentDist = 0.0;
      for (int k = 0; k < dimensionality; k++){
         currentDist += midpoint[k]*midpoint[k];
      }
      double nextDist = sqrt(1 - currentDist);
      
      for (int k = 0; k < dimensionality; k++){
         N[k + i*dimensionality] = midpoint[k];
         if (k == i-1){
            N[k + i*dimensionality] = nextDist;
         }
      }
      
      // restet midpoint to 0
      for (int j = 0; j < dimensionality; j++){
         midpoint[j] = 0.0;
      }
   }
}

/*int __getdelim (char** lineptr, size_t* n, int terminator,FILE* stream)
{
    int c;
    size_t len = 0;
    size_t linesize = 0;
    const size_t BLOCKSIZE = 255;
    if( lineptr == NULL || stream == NULL || n == NULL)
    {
        _set_errno(EINVAL);
        return -1;
    }
    linesize = BLOCKSIZE;
    *lineptr = (char*) malloc(BLOCKSIZE);
    while( (c = fgetc(stream)) != EOF && c != terminator)
    {
        if( (len+1) == linesize)
        {
            linesize += BLOCKSIZE;
            *lineptr = (char*) realloc(*lineptr, linesize);
        }
        (*lineptr)[len++] = c;
    }
    if( len == 0 && c != terminator) // check for blank lines
    {
        _set_errno(EINVAL);
        free(*lineptr);
        *lineptr = NULL;
        return -1;
    }
    (*lineptr)[len] = 0; // null-terminate the string
        
    return len;
}*/

int main(int argc, char *argv[]){
   printf("first line \n");
   fflush(stdout);
   printf("h");
   fflush(stdout);
   printf("argc: %d\n", argc);
   fflush(stdout);

   int maxID = 0;
   int id1 = 0;
   int id2 = 0;
   double id3 = 0;
   
   //graphs\\pam\\graph_n_1000_m_1_t_1.txt
   //graphs\\pam\\graph_n_32_m_1_t_1.txt
   // Aug23Update\\smallgraph.txt
   //graphs\\gnp\\graph_n_1000_p_2.txt
   FILE* NSaveFP = NULL;
   std::string fileString = "";
   printf("argc: %d\n", argc);
   fflush(stdout);
   if (argc >= 3){
      printf("here\n");
      fflush(stdout);
      printf("argv1: %s\n",argv[1]);
      fflush(stdout);
      printf("argv2: %s\n",argv[2]);
      fflush(stdout);
      NSaveFP = fopen(argv[2], "wb");
      if(NSaveFP == NULL){
         perror("fopen");
      }
      printf("fopen, NSaveFP: %d\n", NSaveFP);
      fflush(stdout);
      fileString = argv[1];
      printf("fileString\n");
      fflush(stdout);
   } else {
      NSaveFP = fopen("C:\\Users\\Peter Oostema\\Documents\\school\\graphResearch\\Mar312020-20200401T020813Z-001\\Mar312020\\graphEmbed\\graphs\\temp\\embedP1.bin", "wb");
      //fileString = "C:\\Users\\Peter Oostema\\Documents\\school\\graphResearch\\Mar312020-20200401T020813Z-001\\Mar312020\\graphEmbed\\graphs\\pam\\graph_n_1000_m_1_t_1.txt";
      fileString = "C:\\Users\\Peter Oostema\\Documents\\school\\graphResearch\\Mar312020-20200401T020813Z-001\\Mar312020\\graphEmbed\\graphs\\pam\\graph_n_32_m_2_t_1.txt";
   }
   cout << "fileString: " << fileString << "\n";
   std::ifstream file(fileString);
   if (!file.is_open()){
      printf("file not open\n");
      cout << fileString << endl;
      cerr << "Error: " << strerror(errno);
   }
   printf("ifstream\n");
   fflush(stdout);
   std::string line;

   size_t pos = 0;   
   std::string delimiter = "\t";
   std::string endLineDelim = "\r\n";
   std::string token;
   /*printf("getlinebf\n");
   fflush(stdout);
   while(std::getline(file, line)){
      //printf("getlineat\n");
      //fflush(stdout);
      if ((pos = line.find(delimiter)) != std::string::npos) {
         token = line.substr(0, pos);
         id1 = stoi(token);
         line.erase(0, pos + delimiter.length());
      }
      id2 = stoi(line);
      if (id1 > maxID){
         maxID = id1;
      }
      if (id2 > maxID){
         maxID = id2;
      }
      //printf("id1, id2: %d, %d\n", id1, id2);
      //printf("maxID: %d\n", maxID);
   }
   printf("maxID: %d", maxID);
   fflush(stdout);*/

   std::ifstream inputFile(fileString); // Replace "input.txt" with your file name
    //int maxID = 0;
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open the input file." << std::endl;
        return 1;
    }
    //std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        int num1, num2;
	double num3;
        // Try to extract three numbers from the line
        if (iss >> num1 >> num2 >> num3) {
            // Process the three numbers as needed
            //std::cout << "Number 1: " << num1 << ", Number 2: " << num2 << ", Number 3: " << num3 << std::endl;
        } else {
            std::cerr << "Invalid line format: " << line << std::endl;
        }
	if (num1 > maxID){
	   maxID = num1;
	}
	if (num2 > maxID){
	   maxID = num2;
	}
    }
    printf("maxID: %d\n", maxID);
    //inputFile.close();

   // read in dynamic edges
   std::string dynamicFileString = "C:\\Users\\Peter Oostema\\Documents\\school\\graphResearch\\Mar312020-20200401T020813Z-001\\Mar312020\\graphEmbed\\graphs\\dynamic\\streamingEdge_lowOverlap_highBlockSizeVar_1000_nodes_2.tsv";
   std::ifstream dyfile(dynamicFileString);
   const unsigned numDynamicEdges = 833;
   int dynamicEdges[numDynamicEdges][2];
   unsigned counter = 0;
   while(std::getline(dyfile, line)){
      //printf("getlineat\n");
      //fflush(stdout);
      if ((pos = line.find(delimiter)) != std::string::npos) {
         token = line.substr(0, pos);
         id1 = stoi(token);
         line.erase(0, pos + delimiter.length());
      }
      if ((pos = line.find(delimiter)) != std::string::npos) {
         token = line.substr(0, pos);
         id2 = stoi(token);
         line.erase(0, pos + delimiter.length());
      }
      dynamicEdges[counter][0] = id1;
      dynamicEdges[counter][1] = id2;
      counter++;
      //printf("id1, id2: %d, %d\n", id1, id2);
      //printf("maxID: %d\n", maxID);
   }
   dyfile.close();

   
   int dimensionality = maxID;
   int n = maxID + 1;
   int nActual = n;
   if ((n % 8) != 0){
      n += 8 - (n % 8);
   }
   // dimensionality = 3;//8;//n - 1;
   dimensionality = n - 1;

   //rewind(file);
   file.clear();
   file.seekg(0, ios::beg);
   
   //double* W = (double*) malloc(n*n*sizeof(double));
   //double* N = (double*) malloc(n*dimensionality*sizeof(double));
   double* W;
   double* N;
   cudaMallocHost((double **) &W, n*n*sizeof(double));
   cudaMallocHost((double **) &N, n*dimensionality*sizeof(double));
   double* distSum = NULL;
   cudaMalloc((void**)&distSum, n*sizeof(double));
   cudaMemset(distSum, 0.0, n);
   double* distSumHost = NULL;
   cudaMallocHost((double **) &distSumHost, n*sizeof(double));


   for (int i = 0; i < n; i++){
      for (int j = 0; j < n; j++){
         W[j + i*n] = NAN;
      }
   }
   
   //printf("ad mat\n");
   /*while(std::getline(file, line)){
      //cout << line << endl <<  flush;
      //std::cout << line << endl;//printf("%s,", line);
      if ((pos = line.find(delimiter)) != std::string::npos) {
         token = line.substr(0, pos);
         id1 = stoi(token);
         line.erase(0, pos + delimiter.length());
      }
      //std::cout << line << endl;//printf("%s,", line);
      if ((pos = line.find(delimiter)) != std::string::npos) {
         token = line.substr(0, pos);
         id2 = stoi(token);
         line.erase(0, pos + delimiter.length());
      }
      id3 = stod(line);
      //printf("%s", line);*/
      /*if ((pos = line.find(endLineDelim)) != std::string::npos) {
         token = line.substr(0, pos);
         id2 = stoi(token);
         line.erase(0, pos + delimiter.length());
         printf("pos: %zd\n", pos);
      }*/
      /* //printf("id1, id2: %d, %d\n", id1, id2);
      W[id2 + id1*n] = id3;
      W[id1 + id2*n] = id3;
      printf("%d, %d, %lf\n", id1, id2, id3);
   }*/

    inputFile.clear(); // Clear any error flags
    inputFile.seekg(0, inputFile.beg);
    //std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        int num1, num2;
	double num3;
        // Try to extract three numbers from the line
        if (iss >> num1 >> num2 >> num3) {
            // Process the three numbers as needed
            //std::cout << "Number 1: " << num1 << ", Number 2: " << num2 << ", Number 3: " << num3 << std::endl;
        } else {
            std::cerr << "Invalid line format: " << line << std::endl;
        }
	W[num1 + num2*n] = num3;
	W[num2 + num1*n] = num3;
	printf("%d, %d, %f\n", num1, num2, num3);
    }

   printf("setupSimplex\n");
   
   setupSimplex(N, n, dimensionality);
   //int dimCutoff = 160;
   //setupLowDim(N, n, dimensionality, dimCutoff);
   // int dimCutoff = 3;
   // setupManifold(N, n, nActual, dimensionality, dimCutoff);
   
//    printf("N\n");
//    for (int i = 0; i < n; i++){
//       for (int j = 0; j < dimensionality; j++){
//           printf("%lf ", N[i*dimensionality + j]);
//       }
//       printf("\n");
//    }
   // printf("W\n");
   // for (int i = 0; i < n; i++){
   //    for (int j = 0; j < n; j++){
   //        printf("%lf ", W[i*n + j]);
   //    }
   //    printf("\n");
   // }
           
   double* NSingleDimSave; // = (double*) malloc(n*dimensionality*sizeof(double));
   double* errors = (double*) calloc(dimensionality, sizeof(double));
   double* energies = (double*) calloc(dimensionality, sizeof(double));
   double* maxEdgeErrors = (double*) calloc(dimensionality, sizeof(double));
   double* Ntmp;

   std::clock_t start;
   double duration;

   int rangeBot = 0;
   int rangeTop = dimensionality;
   int prevI = dimensionality;
   int targetDim = 20;
   int bestDim = 99999999;
   double maxEdgeError = 0.0;
   for (int i = dimensionality; i > 0; i -= 0){
      printf("optC, i: %d\n", i);
      start = std::clock();
      double fit = optimizeC(N, W, dimensionality, n, errors, energies, maxEdgeErrors, i, distSum);
      if (maxEdgeError == 0.0){
         maxEdgeError = fit;
      }
      printf("maxEdgeError, fit: %lf, %lf \n", maxEdgeError, fit);
      duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
      std::cout<<"time: "<< duration <<'\n';
      // printf("fit: %s\n", fit ? "true" : "false");
      //break; ///////////////////////////////////////////
      printf("i: %d\n", i);
      // if (i != targetDim){
      //    i = targetDim;
      // }
      // else {
      //    break;
      // }

      bool fitDim = fit < maxEdgeError * 1.1;
      if (fitDim){
         rangeTop = i;
         if (i < bestDim){
            NSingleDimSave = N;
            bestDim = i;
         }
      } else{
         N = NSingleDimSave;
         rangeBot = i;
      }
      int range = rangeTop - rangeBot;
      printf("range, rb, rt: %d, %d, %d\n", range, rangeBot, rangeTop); fflush(stdout);
      if (fitDim){
         printf("range/2: %d\n", range/2); fflush(stdout);
         i -= range/2;
         //rangeTop = i;
      } else{
         printf("range/2: %d\n", range/2); fflush(stdout);
         i += range/2;
         //rangeBot = i;
      }
      printf("ibfi: %d\n", i); fflush(stdout);
      if ((rangeBot == i) || (rangeTop == i)){
         printf("breaking");
         break;
      }
      // if (range % 2 == 1){
      //    if (fit){
      //       i--;
      //    } else{
      //       i++;
      //    }
      // }
      prevI = i;
//       // use when right dim found
//       int rightDim = 20;
//       if (i < dimensionality){
//          break;
//       }
//       i = rightDim;
      // i = targetDim;
      printf("iati: %d\n", i); fflush(stdout);
   }

   // unsigned numDynamicEdgesTEMP = 21;
   // double* dynamicDisplacements = (double*) malloc(n*numDynamicEdgesTEMP*sizeof(double));
   // for (int i = 0; i <= numDynamicEdgesTEMP; i++){
   //    bool fit = optimizeC(N, W, dimensionality, n, errors, energies, maxEdgeErrors, dimensionality, distSum);
   //    cudaMemcpy(distSumHost, distSum, n*sizeof(double), cudaMemcpyDeviceToHost);
   //    cudaMemset(distSum, 0.0, n*sizeof(double));
   //    if (i < numDynamicEdgesTEMP){
   //       W[dynamicEdges[i][0]*n + dynamicEdges[i][1] ] = 1.0;
   //       W[dynamicEdges[i][1]*n + dynamicEdges[i][0] ] = 1.0;
   //    }
   //    if (i > 0) {
   //       for (int j = 0; j < n; j++){
   //          dynamicDisplacements[(i-1)*n + j] = distSumHost[j];
   //          //distSumHost[j] = 0.0;
   //       }
   //    }
   // }
   // FILE* dynamicDistFP = fopen("dynamicDistplacements.bin", "wb");
   // fwrite(&dynamicDisplacements[0], sizeof(double), n*numDynamicEdgesTEMP, dynamicDistFP);
   // fclose(dynamicDistFP);
   
           
    //FILE* NSaveFP = fopen("NSave.txt", "w+");
   FILE* errorsFP = fopen("errors.txt", "w+");
   FILE* energiesFP = fopen("energies.txt", "w+");
   FILE* meeFP = fopen("maxEdgeError.txt", "w+");
   FILE* distsFP = fopen("distSums.txt", "w+");
   //fprintf(NSaveFP, "%d, %d\n", dimensionality, n);
   fprintf(errorsFP, "%d, %d\n", dimensionality, nActual);
   fprintf(energiesFP, "%d, %d\n", dimensionality, nActual);
   fprintf(meeFP, "%d, %d\n", dimensionality, nActual);
   fprintf(distsFP, "%d, %d\n", dimensionality, nActual);
//    for (int i = 1; i <= dimensionality; i++){
//        for (int j = 0; j < n; j++){
//           for (int k = 0; k < i; k++){
//              //fprintf(NSaveFP, "%f,", NSave[(i-1)*n*dimensionality + j*dimensionality + k]);
//           }
//           fprintf(NSaveFP, "%f,", NSingleDimSave[(i-1) + j*dimensionality]); 
//           //fprintf(NSaveFP, "\n");
//       }
//       fprintf(NSaveFP, "\n");
//    }
        // write double array to doubles for matlab TODO
//    printf("N after sim\n");
//    for (int i = 0; i < n; i++){
//       for (int j = 0; j < dimensionality; j++){
//           printf("%lf ", N[i*dimensionality + j]);
//       }
//       printf("\n");
//    }
   //size_t fwritten = fwrite(&NSave[0], sizeof(double), dimensionality*dimensionality*n, NSaveFP);
   
   printf("writing to files\n");
   // for (int i = 1; i <= dimensionality; i++){
       //printf("%d, %f\n", i, errors[i]);
       fprintf(errorsFP, "%d, %lf\n", 2, errors[2]);
       fprintf(energiesFP, "%d, %lf\n", 2, energies[2]);
       fprintf(meeFP, "%d, %lf\n", 2, maxEdgeErrors[2]);
       //fprintf(distsFP, "%lf\n", distSumHost[i]);
       fflush(errorsFP);
       fflush(energiesFP);
       fflush(meeFP);
       //fflush(distsFP);

      // for (int j = 0; j < nActual; j++){
         //fwrite(&NSave[0], sizeof(double), dimensionality, NSaveFP);
         fwrite(&N[0], sizeof(double), dimensionality*n, NSaveFP);
         // if (i == 1){
         //    for (int k = 0; k < dimensionality; k++){
         //        printf("%lf ", NSave[(i-1)*n*dimensionality + j*dimensionality + k]);
         //    }
         // }
      // }
      printf("wrote: %d \n", 2);
   // }
   fflush(NSaveFP);
   //printf("fwrite: %zu\n", fwritten);
   printf("N[0]: %lf\n", N[0]);

   for (int i = 0; i < nActual; i++){
      fprintf(distsFP, "%lf\n", distSumHost[i]);
   }
   printf("bestDim, %d \n", bestDim);

   //for (int i = 0; i < n; i++){
   //   for (int j = 0; j < n; j++){
   //      printf("%f, ", W[j + i*n]);
   //   }
   //   printf("\n");
   //}

   fclose(NSaveFP);
   fclose(errorsFP);
   fclose(energiesFP);
   fclose(meeFP);
   fclose(distsFP);
   
   //fclose(fileP);
   file.close();

   cudaFreeHost(N);
   cudaFreeHost(W);
   cudaFreeHost(distSumHost);
   cudaFree(distSum);
   cudaDeviceReset();
}