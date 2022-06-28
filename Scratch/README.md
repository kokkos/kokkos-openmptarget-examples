# Scratch memory test case.
The example uses dynamic shared memory to assign values inside an OpenMP kernel.
The native OpenMPTarget version uses the `llvm_omp_target_dynamic_shared_alloc` call to request shared memory.
Size of the shared memory requested depends on the environment variable `LIBOMPTARGET_SHARED_MEMORY_SIZE`.
Currently only a single scratch size is possible for the entire duration of an application run. 
Requires the latest llvm compiler. Succesfully works with llvm/15 from the mainline branch on June 28th 2022.

