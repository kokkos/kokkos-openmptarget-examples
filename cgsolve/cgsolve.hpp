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

#include <ompx.h>

#include <generate_matrix.hpp>
#include <iostream>

struct cgsolve {
    int N, max_iter;
    double tolerance;
    Kokkos::View<double *> y, x;
    CrsMatrix<Kokkos::DefaultExecutionSpace::memory_space> A;

    cgsolve(int N_, int max_iter_in, double tolerance_in)
        : N(N_), max_iter(max_iter_in), tolerance(tolerance_in) {
        CrsMatrix<Kokkos::HostSpace> h_A = Impl::generate_miniFE_matrix(N);
        Kokkos::View<double *, Kokkos::HostSpace> h_x =
            Impl::generate_miniFE_vector(N);

        Kokkos::View<int64_t *> row_ptr("row_ptr", h_A.row_ptr.extent(0));
        Kokkos::View<int64_t *> col_idx("col_idx", h_A.col_idx.extent(0));
        Kokkos::View<double *> values("values", h_A.values.extent(0));
        A = CrsMatrix<Kokkos::DefaultExecutionSpace::memory_space>(
            row_ptr, col_idx, values, h_A.num_cols());
        x = Kokkos::View<double *>("X", h_x.extent(0));
        y = Kokkos::View<double *>("Y", h_x.extent(0));

        Kokkos::deep_copy(x, h_x);
        Kokkos::deep_copy(A.row_ptr, h_A.row_ptr);
        Kokkos::deep_copy(A.col_idx, h_A.col_idx);
        Kokkos::deep_copy(A.values, h_A.values);
    }

    template <class YType, class AType, class XType>
    double spmv_ompt(YType y, AType A, XType x) {
        int rows_per_team = 16;
        int team_size = 16;
        int64_t nrows = y.extent(0);

        auto row_ptr = A.row_ptr.data();
        auto values = A.values.data();
        auto col_idx = A.col_idx.data();
        auto xp = x.data();
        auto yp = y.data();

        int64_t n = (nrows + rows_per_team - 1) / rows_per_team;

        const int64_t nteams = (n + team_size - 1) / team_size;

        Kokkos::Timer timer;
#pragma omp target teams ompx_bare num_teams(n) thread_limit(team_size) \
    firstprivate(row_ptr, values, col_idx, xp, yp)
        {
            const int blockId_x = ompx::block_id_x();
            const int blockDim_x = ompx::block_dim_x();
            const int gridDim_x = ompx::grid_dim_x();
            const int threadId_x = ompx::thread_id_x();
            for (int64_t i = blockId_x; i < n; i += gridDim_x) {
                const int64_t first_row = i * rows_per_team;
                const int64_t last_row = first_row + rows_per_team < nrows
                                             ? first_row + rows_per_team
                                             : nrows;

                for (int64_t row = first_row + threadId_x; row < last_row;
                     row += blockDim_x) {
                    const int64_t row_start = row_ptr[row];
                    const int64_t row_length = row_ptr[row + 1] - row_start;

                    // In ompx_bare, each thread gets its own y_row and hence a
                    // += on it wont create a race condition. When multi-dim
                    // threads are available, this could go to threadId_x and
                    // the current one can go to threadId_y. Thats what Kokkos
                    // does.
                    double y_row = 0.;
                    // #pragma omp simd reduction(+ : y_row)
                    for (int64_t i = 0; i < row_length; ++i) {
                        y_row +=
                            values[i + row_start] * xp[col_idx[i + row_start]];
                    }
                    yp[row] = y_row;
                }
            }
        }
        double time = timer.seconds();

        return time;
    }

    template <class YType, class XType>
    double dot_ompt(YType y, XType x) {
        double result;
        int n = y.extent(0);
        auto xp = x.data();
        auto yp = y.data();

#pragma omp barrier

#pragma omp target teams distribute parallel for is_device_ptr(xp, yp) \
    reduction(+ : result)
        for (int i = 0; i < n; ++i) {
            result += yp[i] * xp[i];
        }
        return result;
    }

    template <class ZType, class YType, class XType>
    void axpby_ompt(ZType z, double alpha, XType x, double beta, YType y) {
        int64_t n = z.extent(0);
        auto xp = x.data();
        auto yp = y.data();
        auto zp = z.data();

        const int team_size = 32;
        const int nteams = (n + team_size - 1) / team_size;

#pragma omp target teams ompx_bare num_teams(nteams) thread_limit(team_size) \
    firstprivate(zp, yp, xp)
        {
            const int blockId = ompx::block_id_x();
            const int blockDim = ompx::block_dim_x();
            const int threadId = ompx::thread_id_x();

            const int i = blockId * blockDim + threadId;
            zp[i] = alpha * xp[i] + beta * yp[i];
        }
    }

    template <class VType>
    void print_vector(int label, VType v) {
        std::cout << "\n\nPRINT " << v.label() << std::endl << std::endl;

        int myRank = 0;
        Kokkos::parallel_for(
            v.extent(0), KOKKOS_LAMBDA(const int i) {
                printf("%i %i %i %lf\n", label, myRank, i, v(i));
            });
        Kokkos::fence();
        std::cout << "\n\nPRINT DONE " << v.label() << std::endl << std::endl;
    }

    template <class VType, class AType>
    int cg_solve_ompt(VType y, AType A, VType b, int max_iter,
                      double tolerance) {
        int myproc = 0;
        int num_iters = 0;

        double normr = 0;
        double rtrans = 0;
        double oldrtrans = 0;

        double spmv_time = 0.;

        int64_t print_freq = max_iter / 10;
        if (print_freq > 50) print_freq = 50;
        if (print_freq < 1) print_freq = 1;
        VType x("x", b.extent(0));
        VType r("r", x.extent(0));
        VType r_("r_", x.extent(0));
        VType p("r", x.extent(0));  // Needs to be global
        VType Ap("r", x.extent(0));
        double one = 1.0;
        double zero = 0.0;
        axpby_ompt(p, one, x, zero, x);

        spmv_time += spmv_ompt(Ap, A, p);
        axpby_ompt(r, one, b, -one, Ap);

        rtrans = dot_ompt(r, r);

        normr = std::sqrt(rtrans);

        if (myproc == 0) {
            std::cout << "Initial Residual = " << normr << std::endl;
        }

        double brkdown_tol = std::numeric_limits<double>::epsilon();

        for (int64_t k = 1; k <= max_iter && normr > tolerance; ++k) {
            if (k == 1) {
                axpby_ompt(p, one, r, zero, r);
            } else {
                oldrtrans = rtrans;
                rtrans = dot_ompt(r, r);
                double beta = rtrans / oldrtrans;
                axpby_ompt(p, one, r, beta, p);
            }

            normr = std::sqrt(rtrans);

            double alpha = 0;
            double p_ap_dot = 0;

            spmv_time += spmv_ompt(Ap, A, p);

            p_ap_dot = dot_ompt(Ap, p);

            if (p_ap_dot < brkdown_tol) {
                if (p_ap_dot < 0) {
                    std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"
                              << std::endl;
                    return num_iters;
                } else
                    brkdown_tol = 0.1 * p_ap_dot;
            }
            alpha = rtrans / p_ap_dot;

            axpby_ompt(x, one, x, alpha, p);
            axpby_ompt(r, one, r, -alpha, Ap);
            num_iters = k;
        }

        // Compute SPMV Bytes and Flops
        double spmv_bytes =
            A.num_rows() * sizeof(int64_t) + A.nnz() * sizeof(int64_t) +
            A.nnz() * sizeof(double) + A.nnz() * sizeof(double) +
            A.num_rows() * sizeof(double);
        double spmv_flops = A.nnz() * 2;

        double GB = (spmv_bytes) / 1024 / 1024 / 1024;

        printf("************ SPMV ************ \n");
        printf("OMPT : SPMV : Data Transfered = %f GBs\n", GB);
        printf("OMPT: SPMV Performance: %lf GFlop/s %lf GB/s \n",
               1e-9 * (spmv_flops * (num_iters + 1)) / spmv_time,
               (1.0 / 1024 / 1024 / 1024) * (spmv_bytes * (num_iters + 1)) /
                   spmv_time);
        return num_iters;
    }

    void run_ompt_test() {
        Kokkos::Timer timer;
        int num_iters = cg_solve_ompt(y, A, x, max_iter, tolerance);
        double time = timer.seconds();

        // Compute Bytes and Flops
        double spmv_bytes =
            A.num_rows() * sizeof(int64_t) + A.nnz() * sizeof(int64_t) +
            A.nnz() * sizeof(double) + A.nnz() * sizeof(double) +
            A.num_rows() * sizeof(double);

        double dot_bytes = x.extent(0) * sizeof(double) * 2;
        double axpby_bytes = x.extent(0) * sizeof(double) * 3;

        double spmv_flops = A.nnz() * 2;
        double dot_flops = x.extent(0) * 2;
        double axpby_flops = x.extent(0) * 3;

        int spmv_calls = 1 + num_iters;
        int dot_calls = num_iters;
        int axpby_calls = 2 + num_iters * 3;

        // OMPT info
        printf("CGSolve \n");
        printf("OMPT: CGSolve for 3D (%i %i %i); %i iterations; %lf time\n", N,
               N, N, num_iters, time);
        printf(
            "OMPT: Performance: %lf GFlop/s %lf GB/s (Calls SPMV: %i Dot: %i "
            "AXPBY: %i\n",
            1e-9 *
                (spmv_flops * spmv_calls + dot_flops * dot_calls +
                 axpby_flops * axpby_calls) /
                time,
            (1.0 / 1024 / 1024 / 1024) *
                (spmv_bytes * spmv_calls + dot_bytes * dot_calls +
                 axpby_bytes * axpby_calls) /
                time,
            spmv_calls, dot_calls, axpby_calls);
    }

    void run_test() {
        printf("*******OpenMPTarget***************\n");
        run_ompt_test();
        printf("\n");
    }
};

