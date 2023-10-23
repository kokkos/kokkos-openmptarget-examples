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
#include <random>

#define debug 0

struct Matvec {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = team_policy::member_type;
  using vector_t = Kokkos::View<int64_t **, Kokkos::LayoutRight, ExecSpace>;
  using matrix_t = Kokkos::View<int64_t ***, Kokkos::LayoutRight, ExecSpace>;
  int64_t N;
  vector_t x, y;
  matrix_t m;

  void init() {
    Kokkos::Timer timer;
    auto h_m = create_mirror_view(Kokkos::HostSpace(), m);
    auto h_x = create_mirror_view(Kokkos::HostSpace(), x);
    auto h_y = create_mirror_view(Kokkos::HostSpace(), y);
    //
    // Use default_random_engine object to introduce randomness.
    std::default_random_engine generator;
    // Initialize uniform_int_distribution class.
    std::uniform_int_distribution<int64_t> distribution(0, N);

    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        for (int k = 0; k < N; ++k)
          h_m(i, j, k) = ((i + 1) * (j + 1) * (k + 1)) % INT64_MAX;

    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        h_x(i, j) = ((i + 1) * (j + 1) * distribution(generator)) % INT_MAX;
        h_y(i, j) = 0;
      }

    Kokkos::deep_copy(m, h_m);
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(y, h_y);

    printf("Init: Timer taken = %f[secs] \n", timer.seconds());
  }

  Matvec(int64_t N_)
      : N(N_), y(vector_t("view-y", N, N)), x(vector_t("view-x", N, N)),
        m(matrix_t("view-m", N, N, N)) {
    init();
  }

  void vector_ompt(int64_t *m_ptr, int64_t *x_ptr, int64_t &y) {
    int64_t result = 0;
#pragma omp simd reduction(+ : result)
    for (int k = 0; k < N; ++k) {
      result += m_ptr[k] * x_ptr[k];
    }
    y += result;
  }

  void team_matvec_ompt(int64_t *m_ptr, int64_t *x_ptr, int64_t *y_ptr) {
#pragma omp for
    for (int j = 0; j < N; ++j) {
      vector_ompt(m_ptr + j * N, x_ptr, y_ptr[j]);
    }
  }

  void batched_matrix_vector_ompt() {
    int64_t *m_ptr = m.data();
    int64_t *x_ptr = x.data();
    int64_t *y_ptr = y.data();
#pragma omp target teams distribute is_device_ptr(m_ptr, x_ptr, y_ptr)
    for (int i = 0; i < N; ++i) {
      {
#pragma omp parallel
        { team_matvec_ompt(m_ptr + i * N * N, x_ptr + i * N, y_ptr + i * N); }
      }
    }
  }

  void run_matvec_ompt(int R) {
    Kokkos::Timer timer;

    for (int r = 0; r < R; ++r) {
      batched_matrix_vector_ompt();
    }

    printf("OMPT: Timer taken = %f[secs] \n", timer.seconds());
  }

  KOKKOS_INLINE_FUNCTION
  void batched_matrix_vector_kokkos() {
    team_policy policy(m.extent(0), 32, 32);

    Kokkos::parallel_for(
        policy, KOKKOS_CLASS_LAMBDA(const member_type &team) {
          const int i = team.league_rank();
          auto y_sub = Kokkos::subview(y, i, Kokkos::ALL);
          auto x_sub = Kokkos::subview(x, i, Kokkos::ALL);
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, N), [&](const int j) {
                auto m_sub = Kokkos::subview(m, i, j, Kokkos::ALL);
                int64_t result = 0;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, N),
                    [&](const int k, int64_t &update) {
                      update += m_sub(k) * x_sub(k);
                    },
                    result);

                y_sub(j) += result;
              });
        });
  }

  KOKKOS_INLINE_FUNCTION
  void run_matvec_kokkos(int R) {
    Kokkos::Timer timer;

    for (int r = 0; r < R; ++r) {
      batched_matrix_vector_kokkos();
    }

    printf("KK: Timer taken = %f[secs] \n", timer.seconds());
  }

  void warmup_ompt() {
    int64_t *m_ptr = m.data();
    int64_t *x_ptr = x.data();
    int64_t *y_ptr = y.data();
#pragma omp target teams distribute is_device_ptr(m_ptr, x_ptr, y_ptr)
    for (int i = 0; i < N; ++i) {
      {
#pragma omp parallel
        { team_matvec_ompt(m_ptr + i * N * N, x_ptr + i * N, y_ptr + i * N); }
      }
    }
  }

  void warmup_kk() {
    team_policy policy(m.extent(0), 32, 32);
    Kokkos::parallel_for(
        policy, KOKKOS_CLASS_LAMBDA(const member_type &team) {
          const int i = team.league_rank();
          auto m_sub = Kokkos::subview(m, i, Kokkos::ALL, Kokkos::ALL);
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N),
                               [&](const int j) {
                                 int64_t result = 0;
                                 Kokkos::parallel_reduce(
                                     Kokkos::ThreadVectorRange(team, N),
                                     [&](const int k, int64_t &update) {
                                       update += m(i, j, k) * x(i, k);
                                     },
                                     result);
                                 y(i, j) += result;
                               });
        });
  }

  void run_test(int R) {

    // OMPT
//    warmup_ompt();
//    auto y_ompt = create_mirror_view(Kokkos::HostSpace(), y);
//    for (int i = 0; i < N; ++i)
 //     for (int j = 0; j < N; ++j)
 //       y_ompt(i, j) = 0;
  //  Kokkos::deep_copy(y, y_ompt);
  //  run_matvec_ompt(R);
  //  Kokkos::deep_copy(y_ompt, y);

    // Kokkos
    warmup_kk();
    auto y_kk = create_mirror_view(Kokkos::HostSpace(), y);
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        y_kk(i, j) = 0;
    Kokkos::deep_copy(y, y_kk);
    run_matvec_kokkos(R);
    Kokkos::deep_copy(y_kk, y);

    // Correctness : check whether OMPT and Kokkos results are the same.
    //for (int i = 0; i < N; ++i)
     // for (int j = 0; j < N; ++j)
      //  if (y_ompt(i, j) != y_kk(i, j))
       //   printf("Error: y(%d,%d): KK = %lu, OMPT = %lu\n", i, j, y_kk(i, j),
        //         y_ompt(i, j));
  }

#if debug
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      printf("y(%d,%d) = %lu\n", i, j, y_ompt(i, j));
#endif
};
