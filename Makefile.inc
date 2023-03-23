KOKKOS_PATH=$(SCRATCH)/Kokkos/kokkos-4.0
SRC = $(wildcard *.cpp)
KOKKOS_CXX_STANDARD=c++17

OFFLOAD_FLAGS =

ifeq ($(arch), A100)
    KOKKOS_ARCH=Ampere80
else ifeq ($(arch), MI250x)
    KOKKOS_ARCH=Vega90A
endif

ifeq ($(backend), cuda)
    CXX = $(KOKKOS_PATH)/bin/nvcc_wrapper
    KOKKOS_CUDA_OPTIONS += "enable_lambda"
    KOKKOS_DEVICES = "Cuda"
else ifeq ($(backend), hip)
    CXX = hipcc
    KOKKOS_DEVICES = "HIP"
else ifeq ($(backend), ompt)
    KOKKOS_DEVICES = "OpenMPTarget"
    OFFLOAD_FLAGS = -fopenmp
    ifeq ($(comp), clang)
        CXX = clang++
        ifeq ($(arch), A100)
            OFFLOAD_FLAGS += -Wno-unknown-cuda-version -Wno-undefined-internal -Wno-pass-failed -Wno-openmp-mapping -Wno-extra-tokens
        else ifeq ($(arch), MI250x)
            OFFLOAD_FLAGS += --offload-arch=gfx90a
        endif
    else ifeq ($(comp), nvhpc)
        KOKKOS_OPTIONS += KOKKOS_INTERNAL_DISABLE_DESUL_ATOMICS
        CXX = nvc++
        OFFLOAD_FLAGS = -mp=gpu -gpu=cc80
    else ifeq ($(comp), rocmclang)
        CXX = amdclang++
        OFFLOAD_FLAGS += -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a
    endif
endif