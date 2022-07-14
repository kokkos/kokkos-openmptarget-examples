#include <Kokkos_Core.hpp>

struct InitializeATag {};
struct InitializeCTag {};
struct RangePolicyScanTag {};
struct TeamPolicyScanTag {};

template <class DataType>
struct SCAN {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
    using member_type = TeamPolicy::member_type;
    const int team_size = 32;
    int league_size;

    using view_1D = Kokkos::View<DataType*, ExecSpace>;
    using view_2D = Kokkos::View<DataType**, ExecSpace>;

    int N;
    view_1D a, b;
    view_2D c, d;

    SCAN(int N_) : N(N_), a(view_1D("a", N)), b(view_1D("b", N)) {
        league_size = N / team_size + 1;
        c = (view_2D("c", league_size, team_size));
        d = (view_2D("d", league_size, team_size));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(InitializeATag, int i) const { a(i) = i + 1; }

    KOKKOS_INLINE_FUNCTION
    void operator()(InitializeCTag, const member_type& team) const {
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, team_size),
            [&](const int i) { c(team.league_rank(), i) = i + 2; });
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(RangePolicyScanTag, int i, DataType& update,
                    bool final) const {
        const DataType val_i = a(i);
        if (final) b(i) = update;

        update += val_i;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TeamPolicyScanTag, const member_type& team) const {
        Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, team_size),
                              [&](const int i, DataType& update, bool final) {
                                  const DataType val_i =
                                      c(team.league_rank(), i);
                                  if (final) d(team.league_rank(), i) = update;

                                  update += val_i;
                              });
    }

    void range_policy_scan() {
        // Initialize the 'a' view.
        Kokkos::parallel_for(
            "Initialize_a",
            Kokkos::RangePolicy<ExecSpace, InitializeATag>(0, N), *this);
        Kokkos::fence();

        Kokkos::Timer timer;
        Kokkos::parallel_scan(
            "RangePolicy Scan",
            Kokkos::RangePolicy<ExecSpace, RangePolicyScanTag>(0, N), *this);
        Kokkos::fence();
        double time = timer.seconds();

        // RangePolicy Scan correctness test
        auto h_a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
        auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
        DataType check = 0.;
        bool pass = true;
        for (int i = 0; i < N; ++i) {
            if (check != h_b(i)) {
                printf(
                    "RangePolicy: Failed correctness: expected check = %d, got a(%d) = %d\n",
                    check, i, h_b(i));
                pass = false;
            }
            check += h_a(i);
        }
        if (pass)
            printf("RangePolicy correctness passed with T[secs] = %f \n", time);
    }

    void team_policy_scan() {
        Kokkos::TeamPolicy<ExecSpace, InitializeCTag> policy_initialize(
            league_size, team_size);
        Kokkos::parallel_for("Initialize_c", policy_initialize, *this);
        Kokkos::fence();

        Kokkos::TeamPolicy<ExecSpace, TeamPolicyScanTag> policy_scan(
            league_size, team_size);
        Kokkos::Timer timer;
        Kokkos::parallel_for("TeamPolicy Scan", policy_scan, *this);
        Kokkos::fence();
        double time = timer.seconds();

        // TeamPolicy Scan correctness test
        auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
        auto h_d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
        DataType check = 0.;
        bool pass = true;
        for (int i = 0; i < league_size; ++i) {
            check = 0;
            for (int j = 0; j < team_size; ++j) {
                if (check != h_d(i, j)) {
                    printf(
                        "TeamPolicy: Failed correctness: expected check = %d, got d(%d,%d) "
                        "= %d\n",
                        check, i, j, h_d(i, j));
                    pass = false;
                }
                check += h_c(i, j);
            }
        }
        if (pass)
            printf("TeamPolicy correctness passed with T[secs] = %f\n", time);
    }

    void run_test() {
        range_policy_scan();
        team_policy_scan();
    }
};

