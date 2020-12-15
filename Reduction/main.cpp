#include <Kokkos_Core.hpp>
#include <cmath>
#include <reduction.hpp>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int64_t N = argc > 1 ? atoi(argv[1]) : 10000;
    int R = argc > 2 ? atoi(argv[2]) : 10;

    Reduction red(N);
    red.run_test(R);

  }
  Kokkos::finalize();
}
