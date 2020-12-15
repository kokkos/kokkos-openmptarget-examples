#include <Kokkos_Core.hpp>
#include <cmath>
#include <string.h>

#define nT 32

struct Reduction {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
  using view_t = Kokkos::View<int64_t *, ExecSpace>;
  using h_view_t = Kokkos::View<int64_t *, HostExecSpace>;
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = team_policy::member_type;

  int64_t N;
  int64_t *scratch;
  int64_t scalar;
  h_view_t h_vector;
  view_t vector;

  Reduction(int64_t N_)
      : N(N_), h_vector(h_view_t("h_vector", N_)), vector(view_t("vector", N_)),
        scalar(0) {
    size_t scratch_size = N * nT * sizeof(int64_t);
    scratch = static_cast<int64_t *>(
        omp_target_alloc(scratch_size, omp_get_default_device()));
  }

  void correctness(char const *str, int R) {
    int64_t result = 0;
    for (int64_t i = 0; i < N; ++i)
      result += h_vector(i);

    int64_t check = R * (N * N * (N + 1)) / 2;
    if (scalar != check) {
      printf("%s : Scalar reduction failure. expected = %lu, result = %lu \n",
             str, check, scalar);
    }

    if (result != check) {
      printf("Vector reduction failure. expected = %lu, result = %lu \n", check,
             result);
    }
  }

  double run_ompt_scalar_redn(int R) {

    // warmup
#pragma omp target teams distribute reduction(+ : scalar)
    for (int i = 0; i < N; ++i) {
#pragma omp parallel for reduction(+ : scalar)
      for (int j = 0; j < N; ++j) {
        scalar += i + 1;
      } // end j
    }

    scalar = 0;
    Kokkos::Timer timer;

    for (int r = 0; r < R; ++r) {
#pragma omp target teams distribute reduction(+ : scalar) thread_limit(nT)
      for (int i = 0; i < N; ++i) {
#pragma omp parallel for reduction(+ : scalar) num_threads(nT)
        for (int j = 0; j < N; ++j) {
          scalar += i + 1;
        } // end j
      }
    }

    return timer.seconds();
  }

  double run_ompt_vector_redn(int R) {
    int64_t *vp = h_vector.data();
    int64_t *scratch_ = scratch;
    int N_ = N;

    for (int64_t i = 0; i < N_; ++i)
      vp[i] = 0;

      // warmup
#pragma omp target teams distribute thread_limit(nT) map(tofrom                \
                                                         : vp [0:N])           \
    is_device_ptr(scratch_)
    for (int64_t i = 0; i < N_; ++i) {
#pragma omp parallel num_threads(nT)
      {
        int64_t *my_scratch = scratch_ + (omp_get_team_num() * nT);
#pragma omp single
        my_scratch[0] = 0;

#pragma omp for reduction(+ : my_scratch [0:1])
        for (int64_t j = 0; j < N_; ++j) {
          my_scratch[0] += i + 1;
        } // end j
#pragma omp single
        vp[i] += my_scratch[0];
      } // end parallel
    }   // end teams

    for (int64_t i = 0; i < N_; ++i)
      vp[i] = 0;

    Kokkos::Timer timer;

    for (int r = 0; r < R; ++r) {
#pragma omp target teams distribute thread_limit(nT) map(tofrom                \
                                                         : vp [0:N])           \
    is_device_ptr(scratch_)
      for (int64_t i = 0; i < N_; ++i) {
#pragma omp parallel num_threads(nT)
        {
          int64_t *my_scratch = scratch_ + (omp_get_team_num() * nT);
#pragma omp single
          my_scratch[0] = 0;

#pragma omp for reduction(+ : my_scratch [0:1])
          for (int64_t j = 0; j < N_; ++j) {
            my_scratch[0] += i + 1;
          } // end j
#pragma omp single
          vp[i] += my_scratch[0];
        } // end parallel
      }   // end teams
    }

    return timer.seconds();
  }

  void run_ompt_reduction(int R) {
    double time = run_ompt_scalar_redn(R);
    printf("OMPT: scalar reduction = %f [secs]\n", time);

    time = run_ompt_vector_redn(R);
    printf("OMPT: vector reduction = %f [secs]\n", time);

    correctness("OMPT", R);
  }

  double run_kk_scalar_redn(int R) {

    int64_t N_ = N;
    int64_t scalar_ = 0;
    team_policy policy(N_, nT);

    // warmup
    Kokkos::parallel_reduce(
        "case-1", policy,
        KOKKOS_LAMBDA(const member_type &team, int64_t &team_update) {
          const int64_t i = team.league_rank();
          int64_t thread_update = 0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, N_),
              [&](const int64_t j, int64_t &update) { update += i + 1; },
              thread_update);

          team_update += thread_update;
        },
        scalar_);

    scalar = 0;
    Kokkos::Timer timer;

    for (int r = 0; r < R; ++r) {
      scalar_ = 0;
      Kokkos::parallel_reduce(
          "case-1", policy,
          KOKKOS_LAMBDA(const member_type &team, int64_t &team_update) {
            const int64_t i = team.league_rank();
            int64_t thread_update = 0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, N_),
                [&](const int64_t j, int64_t &update) { update += i + 1; },
                thread_update);

            Kokkos::single(Kokkos::PerTeam(team),
                           [&]() { team_update += thread_update; });
          },
          scalar_);
      scalar += scalar_;
    }

    return timer.seconds();
  }

  double run_kk_vector_redn(int R) {
    int64_t N_ = N;
    view_t vector_ = vector;
    team_policy policy(N, Kokkos::AUTO);

    // warmup
    for (int64_t i = 0; i < N_; ++i)
      h_vector(i) = 0;

    Kokkos::parallel_for(
        "Vector_reduction", policy, KOKKOS_LAMBDA(const member_type &team) {
          const int64_t i = team.league_rank();
          int64_t thread_scalar = 0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, N_),
              [&](const int j, int64_t &thread_update) {
                thread_update += i + 1;
              },
              thread_scalar);

          vector_(i) += thread_scalar;
        });

    for (int64_t i = 0; i < N_; ++i)
      h_vector(i) = 0;
    Kokkos::deep_copy(vector, h_vector);

    Kokkos::Timer timer;

    for (int r = 0; r < R; ++r) {
      Kokkos::parallel_for(
          "Vector_reduction", policy, KOKKOS_LAMBDA(const member_type &team) {
            const int64_t i = team.league_rank();
            int64_t thread_scalar = 0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, N_),
                [&](const int64_t j, int64_t &thread_update) {
                  thread_update += i + 1;
                },
                thread_scalar);

            Kokkos::single(Kokkos::PerTeam(team),
                           [&]() { vector_(i) += thread_scalar; });
          });
    }
    Kokkos::deep_copy(h_vector, vector);

    return timer.seconds();
  }

  void run_kokkos_reduction(int R) {
    double time = run_kk_scalar_redn(R);
    printf("KK: scalar reduction = %f [secs]\n", time);

    time = run_kk_vector_redn(R);
    printf("KK: vector reduction = %f [secs]\n", time);

    correctness("KK", R);
  }

  void run_test(int R) {
    run_ompt_reduction(R);
    run_kokkos_reduction(R);
  }
};
