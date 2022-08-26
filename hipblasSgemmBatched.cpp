//
// Wrapper for hipblasSgemm function.
//
// Alan Gray, NVIDIA
//
// Modified for hipblas by Andreas Mueller, ECMWF

#include <stdio.h>
#include <hip/hip_runtime.h>
#include "hipblas.h"

bool alreadyAllocated_sgemm = false;
bool alreadyAllocated_sgemm_handle = false;

float **d_Aarray_sgemm;
float **d_Barray_sgemm;
float **d_Carray_sgemm;

float **Aarray_sgemm;
float **Barray_sgemm;
float **Carray_sgemm;

hipblasHandle_t handle_sgemm;

extern "C" void hipblasSgemmStridedBatched_wrapper(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, long long tda, const float *B, int ldb, long long tdb, float beta, float *C, int ldc, long long tdc, int batchCount)
{

  printf("HIPBLAS m=%d,n=%d,k=%d,batchcount=%d\n", m, n, k, batchCount);

  hipblasOperation_t op_t1 = HIPBLAS_OP_N, op_t2 = HIPBLAS_OP_N;

  if (transa == 'T' || transa == 't')
    op_t1 = HIPBLAS_OP_T;

  if (transb == 'T' || transb == 't')
    op_t2 = HIPBLAS_OP_T;

  if (!alreadyAllocated_sgemm_handle)
  {
    hipblasCreate(&handle_sgemm);
    alreadyAllocated_sgemm_handle = true;
  }
  printf("before sgemm: hipDeviceSynchronize=%d\n", hipDeviceSynchronize());
  if (hipblasSgemmStridedBatched(handle_sgemm, op_t1, op_t2, m, n, k, &alpha, (const float *)A, lda, tda, (const float *)B, ldb, tdb, &beta, (float *)C, ldc, tdc, batchCount) == HIPBLAS_STATUS_SUCCESS)
  {
    printf("HIP in hipblasSgemmBatched.cpp: hipblasSgemmBatched succeeded\n");
  }
  printf("after sgemm: hipDeviceSynchronize=%d\n", hipDeviceSynchronize());
}

extern "C" void hipblasSgemmBatched_finalize()
{

  if (alreadyAllocated_sgemm)
  {

    hipFree(Aarray_sgemm);
    hipFree(Barray_sgemm);
    hipFree(Carray_sgemm);

    hipFree(d_Aarray_sgemm);
    hipFree(d_Barray_sgemm);
    hipFree(d_Carray_sgemm);
  }

  if (alreadyAllocated_sgemm_handle)
  {
    hipblasDestroy(handle_sgemm);
  }
}
