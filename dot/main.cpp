#include <Kokkos_Core.hpp>
#include <cmath>
#include <dot.hpp>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    int R = argc > 2 ? atoi(argv[2]) : 10;
    DOT dot(N, false);
    dot.run_test(R);
  }
  Kokkos::finalize();
}
