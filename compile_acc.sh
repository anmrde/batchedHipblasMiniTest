#!/bin/bash 
module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
CC -std=c++11 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -x hip -c hipblasSgemmBatched.cpp
ftn -c hipblas_mod.F90
ftn -Dacc -c hipblas-batch.F90
ftn -L${ROCM_PATH}/lib -lamdhip64 -L${ROCM_PATH}/hipblas/lib -lhipblas -o hipblas-batch.exe hipblas-batch.o hipblas_mod.o hipblasSgemmBatched.o
