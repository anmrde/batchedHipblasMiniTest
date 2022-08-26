#!/bin/bash 
module load CrayEnv
module load rocm
hipcc --offload-arch=gfx90a -c hipblasSgemmBatched.cpp
amdflang -c hipblas_mod.F90
amdflang -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
        -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -c hipblas-batch.F90
amdflang -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
         -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a \
	 --rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64 \
	 -L${ROCM_PATH}/hipblas/lib -lhipblas -o hipblas-batch.exe \
	 hipblas-batch.o hipblas_mod.o hipblasSgemmBatched.o
