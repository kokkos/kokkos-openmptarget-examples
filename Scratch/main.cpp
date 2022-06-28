#include <Kokkos_Core.hpp>

using ExecSpace = Kokkos::DefaultExecutionSpace;
using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using member_type = TeamPolicy::member_type;
using ScratchSpace = ExecSpace::scratch_memory_space;
using scratch_view_type =
    Kokkos::View<int*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// FIXME - Something similar to this should be in the OpenMPTarget backend of
// Kokkos to request specific amount of scratch memory to the OpenMP runtime.
namespace Kokkos {
namespace Impl {

inline void* get_dynamic_shared() { return NULL; }
#pragma omp begin declare variant match(device = {arch(nvptx64)})
extern "C" void* llvm_omp_target_dynamic_shared_alloc();
inline void* get_dynamic_shared() {
    return llvm_omp_target_dynamic_shared_alloc();
}
#pragma omp end declare variant
}  // namespace Impl
}  // namespace Kokkos

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        const int team_size = 32;
        TeamPolicy policy(1, team_size);
        size_t scratch_size = scratch_view_type::shmem_size(2);
        policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

        Kokkos::View<int[2], ExecSpace> v("v");

#if defined(KOKKOS_ENABLE_OPENMPTARGET)

        int x[2];

#pragma omp target parallel map(from : x[:2])
        {
            int* buf = (int*)Kokkos::Impl::get_dynamic_shared();
#pragma omp barrier
            if (omp_get_thread_num() == 0) {
                buf[0] = 1;
                buf[1] = 2;
            }
#pragma omp barrier
            if (omp_get_thread_num() == 1) {
                x[0] = buf[0] + buf[1];
                x[1] = buf[1] - buf[0];
            }
#pragma omp barrier
        }

        if (x[0] == 3 && x[1] == 1)
            printf("Native OpenMPTarget: Correctness passed \n");
        else {
            printf("Native OpenMPTarget:Failed: x[0] = %d, expected 3\n", x[0]);
            printf("Native OpenMPTarget:Failed: x[1] = %d, expected 1\n", x[1]);
        }
#endif

        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const member_type& team) {
                scratch_view_type scratch_view(team.team_scratch(0), 2);

                if (team.team_rank() == 0) {
                    scratch_view(0) = 1;
                    scratch_view(1) = 2;
                }
                team.team_barrier();
                if (team.team_rank() == 0) {
                    v(0) = scratch_view(0) + scratch_view(1);
                    v(1) = scratch_view(1) - scratch_view(0);
                }
                team.team_barrier();
            });
        Kokkos::fence();

        auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);

        if (h_v(0) == 3 && h_v(1) == 1)
            printf("%s Correctness passed \n", typeid(ExecSpace).name());
        else {
            printf("%s Failed: v(0) = %d, expected 3\n",
                   typeid(ExecSpace).name(), h_v(0));
            printf("%s Failed: v(1) = %d, expected 1\n",
                   typeid(ExecSpace).name(), h_v(1));
        }
    }
    Kokkos::finalize();

    return 0;
}
