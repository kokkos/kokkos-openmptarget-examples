#include <Kokkos_Core.hpp>
#include <axpy.hpp>
#include <cmath>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    int R = argc > 2 ? atoi(argv[2]) : 10;
    AXPBY axpby(N, false);
    axpby.run_test(R);
  }
  Kokkos::finalize();
}
