#include <Kokkos_Core.hpp>

int main(int argc, char** argv)
{
    int n_omp = 0, n_kk = 0;
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
#pragma omp target map(tofrom: n_omp)
    n_omp = getNumberofProcessorElements();
#endif
    printf("OMP: n = %d\n",n_omp);
    Kokkos::initialize();
    {
        n_kk = Kokkos::DefaultExecutionSpace().concurrency();
        printf("KK: n = %d\n",n_kk);
    }
    Kokkos::finalize();

    return 0;
}
