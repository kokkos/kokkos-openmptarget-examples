#include <Kokkos_Core.hpp>
#include <scan.hpp>

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        int N = argc > 1 ? atoi(argv[1]) : 10000;
        using DataType = int;
        SCAN<DataType> scan(N);
        scan.run_test();
    }
    Kokkos::finalize();

    return 0;
}
