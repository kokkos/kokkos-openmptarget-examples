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

struct DOT {
  using view_t = Kokkos::View<double *>;
  int N;
  view_t x, y, z;

  bool fence_all;
  DOT(int N_, bool fence_all_)
      : N(N_), x(view_t("X", N)), y(view_t("Y", N)), fence_all(fence_all_) {}

  KOKKOS_FUNCTION
  void operator()(int i, double &lsum) const { lsum += x(i) * y(i); }

  double kk_dot(int R) {
    // Warmup
    double result;
    Kokkos::parallel_reduce("kk_dot_wup", N, *this, result);
    Kokkos::fence();

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      Kokkos::parallel_reduce("kk_dot", N, *this, result);
    }
    Kokkos::fence();
    double time = timer.seconds();
    return time;
  }

#ifdef KOKKOS_ENABLE_OPENMPTARGET
  double ompt_dot(int R) {
    DOT f(*this);
    int N_ = N;
    // Warmup
    double result = 0.;
#pragma omp target teams distribute parallel for simd data map(to: f, N_) reduction(+:result)
    for (int i = 0; i < N_; i++) {
      f(i, result);
    }
    Kokkos::fence();

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
#pragma omp target teams distribute parallel for simd data map(to: f, N_) reduction(+:result)
      for (int i = 0; i < N_; i++) {
        f(i, result);
      }
      if (fence_all)
        Kokkos::fence();
    }
    Kokkos::fence();
    double time = timer.seconds();
    return time;
  }

  double ompt_raw_dot(int R) {
    double *xp = x.data();
    double *yp = y.data();
    double *zp = z.data();
#if 1
    // without temporary for N its 85GB/s with this its 330GB/s on V100
    int N_ = N;
#else
#define N_ N
#endif
    // Warmup
    double result = 0.;
#pragma omp target teams distribute parallel for \
      simd is_device_ptr(xp,yp) data map(to: N_) reduction(+:result)
    for (int i = 0; i < N_; i++) {
      result += xp[i] * yp[i];
    }
    Kokkos::fence();

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
#pragma omp target teams distribute parallel for \
        simd is_device_ptr(xp,yp,zp) data map(to: N_) reduction(+:result)
      for (int i = 0; i < N_; i++) {
        result += xp[i] * yp[i];
      }
      if (fence_all)
        Kokkos::fence();
    }
    Kokkos::fence();
    double time = timer.seconds();
    return time;
  }
#endif

  void run_test(int R) {
    double bytes_moved = 1. * sizeof(double) * N * 2 * R;
    double GB = bytes_moved / 1024 / 1024 / 1024;
    double time_kk = kk_dot(R);
    printf("DOT KK: %e s %e GB/s\n", time_kk, GB / time_kk);
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    double time_ompt = ompt_dot(R);
    printf("DOT OMPT: %e s %e GB/s\n", time_ompt, GB / time_ompt);
    double time_ompt_raw = ompt_raw_dot(R);
    printf("DOT OMPT_RAW: %e s %e GB/s\n", time_ompt_raw, GB / time_ompt_raw);
#endif
  }
};
