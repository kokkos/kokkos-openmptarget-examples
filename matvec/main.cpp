#include <Kokkos_Core.hpp>
#include <matvec.hpp>
#include <cmath>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int64_t N = argc > 1 ? atoi(argv[1]) : 1000;
    int R = argc > 2 ? atoi(argv[2]) : 10;
    Matvec matvec(N);
    matvec.run_test(R);
  }
  Kokkos::finalize();
}
