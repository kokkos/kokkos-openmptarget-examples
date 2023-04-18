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

#include <limits.h>
#include <string.h>

#include <Kokkos_Core.hpp>
#include <cmath>

using ExecSpace = Kokkos::DefaultExecutionSpace;
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;

struct point {
    int x;
    int y;

    KOKKOS_INLINE_FUNCTION
    point() : x(0), y(0) { ; }

    KOKKOS_INLINE_FUNCTION
    point(int x_, int y_) : x(x_), y(y_) { ; }

    KOKKOS_INLINE_FUNCTION
    point(const point &src) {
        x = src.x;
        y = src.y;
    }

    KOKKOS_INLINE_FUNCTION
    point(const point &&src) {
        x = src.x;
        y = src.y;
    }

    KOKKOS_INLINE_FUNCTION
    point &operator=(const point &src) {
        x = src.x;
        y = src.y;

        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    bool operator>(const point &src) const {
        if (src.x < x && src.y < y) return true;

        return false;
    }

    KOKKOS_INLINE_FUNCTION
    bool operator<(const point &src) const {
        if (src.x > x && src.y > y) return true;

        return false;
    }

    KOKKOS_INLINE_FUNCTION
    point &operator+=(const point &src) {
        x += src.x;
        y += src.y;

        return *this;
    }
};

namespace Kokkos {
template <>
struct reduction_identity<point> {
    KOKKOS_FORCEINLINE_FUNCTION static point sum() { return point(0, 0); }
    KOKKOS_FORCEINLINE_FUNCTION static point min() {
        return point(INT_MAX, INT_MAX);
    }
    KOKKOS_FORCEINLINE_FUNCTION static point max() { return point(0, 0); }
};
}  // namespace Kokkos

struct custom_reduction {
    int N;
    int R;

    using h_view = Kokkos::View<point *, HostExecSpace>;
    using d_view = Kokkos::View<point *, ExecSpace>;

    h_view h_points;
    d_view d_points;

    custom_reduction(int N_, int R_)
        : N(N_),
          R(R_),
          h_points(h_view("h_points", N)),
          d_points(d_view("d_points", N)) {}

    KOKKOS_INLINE_FUNCTION
    static void minproc(struct point *out, struct point *in) {
        if (in->x < out->x) out->x = in->x;
        if (in->y < out->y) out->y = in->y;
    }

    KOKKOS_INLINE_FUNCTION
    static void maxproc(struct point *out, struct point *in) {
        if (in->x > out->x) out->x = in->x;
        if (in->y > out->y) out->y = in->y;
    }

    KOKKOS_INLINE_FUNCTION
    static void addproc(struct point *out, struct point *in) {
        out->x += in->x;
        out->y += in->y;
    }

    void init() {
        for (int i = 0; i < N; ++i) h_points(i) = point(i + 1, i + 2);

        Kokkos::deep_copy(d_points, h_points);
    }

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
#pragma omp declare reduction(min : struct point : minproc(&omp_out, &omp_in)) \
    initializer(omp_priv = {INT_MAX, INT_MAX})

#pragma omp declare reduction(max : struct point : maxproc(&omp_out, &omp_in)) \
    initializer(omp_priv = {0, 0})

#pragma omp declare reduction(add : struct point : addproc(&omp_out, &omp_in)) \
    initializer(omp_priv = {omp_in.x, omp_in.y})

    void find_enclosing_rectangle_omp(int n, struct point *points) {
        struct point minp = {INT_MAX, INT_MAX}, maxp = {0, 0}, sump = {0, 0};
        int i;

#pragma omp target teams distribute parallel for reduction(add : sump) \
    reduction(min : minp) reduction(max : maxp) is_device_ptr(points)
        for (i = 0; i < n; i++) {
            addproc(&sump, &points[i]);
            minproc(&minp, &points[i]);
            maxproc(&maxp, &points[i]);
        }

        printf("Native OpenMP\n");
        printf("min : result = (%d, %d),   expected = (%d, %d)\n", minp.x,
               minp.y, 1, 2);
        printf("max : result = (%d, %d),   expected = (%d, %d)\n", maxp.x,
               maxp.y, N, N + 1);
        printf("sum : result = (%d, %d),   expected = (%d, %d)\n", sump.x,
               sump.y, N * (N + 1) / 2, N * (N + 3) / 2);
        printf("\n");
    }
#endif

    void find_enclosing_rectangle_kk() {
        point minp = point{INT_MAX, INT_MAX}, maxp = point(0, 0), sump = {0, 0};
        Kokkos::RangePolicy<ExecSpace> policy(0, N);

        Kokkos::parallel_reduce(
            "kk_parallel_reduce_sum", policy,
            KOKKOS_CLASS_LAMBDA(const int i, point &lsum) {
                lsum += d_points(i);
            },
            sump);
        Kokkos::parallel_reduce(
            "kk_parallel_reduce_sum", policy,
            KOKKOS_CLASS_LAMBDA(const int i, point &lsum) {
                lsum += d_points(i);
            },
            Kokkos::Sum<point>(sump));

        Kokkos::fence();

        Kokkos::parallel_reduce(
            "kk_parallel_reduce_min", policy,
            KOKKOS_CLASS_LAMBDA(const int i, point &lmin) {
                if (lmin > d_points(i)) lmin = d_points(i);
            },
            Kokkos::Min<point>(minp));

        Kokkos::fence();

        Kokkos::parallel_reduce(
            "kk_parallel_reduce_max", policy,
            KOKKOS_CLASS_LAMBDA(const int i, point &lmax) {
                if (lmax < d_points(i)) lmax = d_points(i);
            },
            Kokkos::Max<point>(maxp));

        Kokkos::fence();

        printf("Kokkos - %s\n", typeid(ExecSpace).name());
        printf("min : result = (%d, %d),   expected = (%d, %d)\n", minp.x,
               minp.y, 1, 2);
        printf("max : result = (%d, %d),   expected = (%d, %d)\n", maxp.x,
               maxp.y, N, N + 1);
        printf("sum : result = (%d, %d),   expected = (%d, %d)\n", sump.x,
               sump.y, N * (N + 1) / 2, N * (N + 3) / 2);
        printf("\n");
    }

    void run() {
        init();
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
        find_enclosing_rectangle_omp(N, d_points.data());
#endif
        find_enclosing_rectangle_kk();
    }
};
