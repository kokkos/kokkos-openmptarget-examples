//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>
#include <cmath>

struct AXPBY {
    using view_t = Kokkos::View<double*>;
    size_t N;
    view_t x, y, z;

    AXPBY(int N_)
        : N(N_), x(view_t("X", N)), y(view_t("Y", N)), z(view_t("Z", N)) {
        Kokkos::deep_copy(x, 1);
        Kokkos::deep_copy(y, 2);
    }

    KOKKOS_FUNCTION
    void operator()(int i) const { z(i) = x(i) + y(i); }

    double kk_axpby(int R) {
        // Warmup
        Kokkos::parallel_for("kk_axpby_wup", N, *this);
        Kokkos::fence();

        Kokkos::Timer timer;
        for (int r = 0; r < R; r++) {
            Kokkos::parallel_for(
                "kk_axpby", Kokkos::RangePolicy<Kokkos::IndexType<int>>(0, N),
                *this);
        }
        Kokkos::fence();
        auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), z);
        double time = timer.seconds();
        return time;
    }

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
    double native_openmp_axpby(int R) {
        double* x_ = x.data();
        double* y_ = y.data();
        double* z_ = z.data();

        // Warmup
#pragma omp target teams distribute parallel for is_device_ptr(x_, y_, z_)
        for (int i = 0; i < N; ++i) {
            z_[i] = x_[i] + y_[i];
        }

        Kokkos::Timer timer;
        for (int r = 0; r < R; r++) {
#pragma omp target teams distribute parallel for is_device_ptr(x_, y_, z_)
            for (int i = 0; i < N; ++i) {
                z_[i] = x_[i] + y_[i];
            }
        }
        double time = timer.seconds();
        return time;
    }

    double lambda_openmp_axpby(int R) {
        double* x_ = x.data();
        double* y_ = y.data();
        double* z_ = z.data();

        auto axpby_lambda = [=](size_t i) { z_[i] = x_[i] + y_[i]; };

        // Warmup
#pragma omp target teams distribute parallel for map(to : axpby_lambda)
        for (int i = 0; i < N; ++i) {
            axpby_lambda(i);
        }

        Kokkos::Timer timer;

        for (int r = 0; r < R; r++) {
#pragma omp target teams distribute parallel for map(to : axpby_lambda)
            for (size_t i = 0; i < N; ++i) axpby_lambda(i);
        }
        double time = timer.seconds();
        return time;
    }
#endif

    void run_test(int R) {
        double bytes_moved = 1. * sizeof(double) * N * 3 * R;
        double GB = bytes_moved / 1024 / 1024 / 1024;

        // AXPBY as Kokkos kernels
        printf("am here \n");
        double time_kk = kk_axpby(R);
        printf("AXPBY KK: %e s %e GiB/s\n", time_kk, GB / time_kk);

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
        // AXPBY as LAMBDA inside OpenMP
        double time_lambda_openmp = lambda_openmp_axpby(R);
        printf("AXPBY lambda-openmp: %e s %e GiB/s\n", time_lambda_openmp,
               GB / time_lambda_openmp);

        // AXPBY as native OpenMP
        double time_native_openmp = native_openmp_axpby(R);
        printf("AXPBY native-openmp: %e s %e GiB/s\n", time_native_openmp,
               GB / time_native_openmp);
#endif
    }
};
