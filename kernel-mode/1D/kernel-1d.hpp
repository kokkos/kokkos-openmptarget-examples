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
#include <string.h>
#include <ompx.h>


//#define check_correctness

struct omp_kernel_1d {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = team_policy::member_type;

  using view_1d = Kokkos::View<int64_t *, ExecSpace>;
  using view_2d = Kokkos::View<int64_t **, ExecSpace>;

  static const int nT = 64;
  int64_t N;
  view_1d vector;

  omp_kernel_1d(int64_t N_)
      : N(N_), vector(view_1d("vector", N_)) {
  }

  #ifdef check_correctness
  
  void correctness_omp_for() {
    int64_t result = 0;
    int64_t N_ = N;

    auto h_vec = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vector);
    for (int64_t i = 0; i < N_; ++i)
      result += h_vec(i);

    printf("result = %lu, expected = %lu \n",result, N_*(N_+1)/2);
  }
  #endif // check_correctness

  double omp_for(int R)
  {
    int64_t N_ = N;

    // warmup
#pragma omp target teams distribute parallel for firstprivate(vector)
    for (int i = 0; i < N_; ++i) {
      vector(i) = i+1;
    }
    
    Kokkos::Timer timer;
    for(int i = 0; i < R; ++i)
    {
#pragma omp target teams distribute parallel for
      for (int i = 0; i < N_; ++i) {
        vector(i) = i+1;
      }
    }

    double time_taken = timer.seconds();

#ifdef check_correctness
    correctness_omp_for();
#endif // check_correctness

    return time_taken;
  }

  double kk_for(int R)
  {
    int64_t N_ = N;

    // warmup
    Kokkos::parallel_for("kk-par-for", N_, KOKKOS_LAMBDA(const int i){
      vector(i) = i+1;
    });
    
    Kokkos::Timer timer;
    for(int i = 0; i < R; ++i)
    {
      Kokkos::parallel_for("kk-par-for", N_, KOKKOS_LAMBDA(const int i){
        vector(i) = i+1;
      });
    }

    double time_taken = timer.seconds();

#ifdef check_correctness
    correctness_omp_for();
#endif // check_correctness

    return time_taken;
  }

  double omp_for_kernel(int R)
  {
    int64_t N_ = N;

    const int nTeams = N_/nT + !! (N_%nT);
    // warmup
#pragma omp target teams ompx_bare num_teams(nTeams,1,1) thread_limit(nT,1,1) firstprivate(vector)
    {
      const int blockIdx  = ompx::block_id(ompx::dim_x);
      const int blockDimx = ompx::block_dim(ompx::dim_x);
      const int threadIdx = ompx::thread_id(ompx::dim_x);

      const int i= blockIdx*blockDimx+threadIdx;

      if(i < N_)
      {
        vector(i) = i+1;
      }
    }
    
    Kokkos::Timer timer;
    for(int i = 0; i < R; ++i)
    {
#pragma omp target teams ompx_bare num_teams(nTeams,1,1) thread_limit(nT,1,1) firstprivate(vector)
      {
          const int blockIdx  = ompx::block_id(ompx::dim_x);
          const int blockDimx = ompx::block_dim(ompx::dim_x);
          const int threadIdx = ompx::thread_id(ompx::dim_x);

          const int i= blockIdx*blockDimx+threadIdx;

          if(i< N_)
          {
            vector(i) = i+1;
          }
      }
    }

    double time_taken = timer.seconds();

#ifdef check_correctness
    correctness_omp_for();
#endif // check_correctness

    return time_taken;
  }

  double omp_reduce(int R)
  {
    int64_t N_ = N;
    int64_t result = 0;

    // warmup
#pragma omp target teams distribute parallel for reduction(+:result)
      for (int i = 0; i < N_; ++i)
        result += vector(i);
    
    Kokkos::Timer timer;
    for(int i = 0; i < R; ++i)
    {
      result = 0;
#pragma omp target teams distribute parallel for reduction(+:result)
      for (int i = 0; i < N_; ++i) {
        result += vector(i);
      }
    }

    double time_taken = timer.seconds();

    int64_t expected = N*(N+1)/2;
    if(result != expected)
      printf("result = %ld, expected = %ld\n",result, expected);


    return time_taken;
  }

  double omp_reduce_kernel(int R)
  {
    int64_t N_ = N;
    int64_t result = 0;

    const int nTeams = N_/nT + !! (N_%nT);
    size_t scratch_size = nT*sizeof(int64_t);
    view_1d partial_results = view_1d("partial_results",nTeams);
    /*int64_t* partial_results = static_cast<int64_t*>(omp_target_alloc(nTeams*sizeof(int64_t), omp_get_default_device()));*/


    // warmup
#pragma omp target teams ompx_bare num_teams(nTeams) thread_limit(nT) ompx_dyn_cgroup_mem(scratch_size) firstprivate(vector, partial_results)
  {
      const int blockidx  = ompx::block_id(ompx::dim_x);
      const int blockdimx = ompx::block_dim(ompx::dim_x);
      const int threadidx = ompx::thread_id(ompx::dim_x);
      int64_t *buf = static_cast<int64_t*>(llvm_omp_target_dynamic_shared_alloc());
      buf[threadidx] = 0;
      ompx_sync_block_acq_rel();

      const int i= blockidx*blockdimx+threadidx;

      if(i < N_)
        buf[threadidx] += vector(i);
      ompx_sync_block_acq_rel();

      if(threadidx == 0)
      {
        partial_results(blockidx) = 0;

        for(int tid = 0; tid < blockdimx; ++tid)
          partial_results(blockidx) += buf[tid];
      }
  }
    
    Kokkos::Timer timer;
/*    for(int i = 0; i < 2; ++i)*/
/*    {*/
/*      result = 0;*/
/**/
/*#pragma omp target teams ompx_bare num_teams(nTeams) thread_limit(nT) ompx_dyn_cgroup_mem(scratch_size) firstprivate(partial_results)*/
/*  {*/
/*      const int blockIdx  = ompx::block_id(ompx::dim_x);*/
/*      const int blockDimx = ompx::block_dim(ompx::dim_x);*/
/*      const int threadIdx = ompx::thread_id(ompx::dim_x);*/
/*      int64_t *buf = static_cast<int64_t*>(llvm_omp_target_dynamic_shared_alloc());*/
/*      buf[threadIdx] = 0;*/
/*      ompx_sync_block_acq_rel();*/
/**/
/*      const int i= blockIdx*blockDimx+threadIdx;*/
/**/
/*      if(i < N_)*/
/*        buf[threadIdx] += vector(i);*/
/*      ompx_sync_block_acq_rel();*/
/**/
/*      if(threadIdx == 0)*/
/*      {*/
/*        partial_results[blockIdx] = 0;*/
/**/
/*        for(int tid = 0; tid < blockDimx; ++tid)*/
/*          partial_results[blockIdx] += buf[tid];*/
/*      }*/
/*  }*/
/**/
/*    auto h_partial_results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), partial_results);*/
/*    for(int i = 0; i < nTeams; ++i)*/
/*      result += h_partial_results[i];*/
/*    }*/


    double time_taken = timer.seconds();

    int64_t expected = N_*(N_+1)/2;
    if(result != expected)
      printf("kernel: result = %ld, expected = %ld\n",result, expected);

    return time_taken;
  }

  double kk_reduce(int R)
  {
    int64_t N_ = N;
    int64_t result = 0;

    // warmup
    Kokkos::parallel_reduce("kk-par-reduce", N_, KOKKOS_LAMBDA(const int i, int64_t& lsum){
      lsum += vector(i);
    }, result);
    
    Kokkos::Timer timer;
    for(int i = 0; i < R; ++i)
    {
      result = 0;
      Kokkos::parallel_reduce("kk-par-reduce", N_, KOKKOS_LAMBDA(const int i, int64_t& lsum){
        lsum += vector(i);
      }, result);
    }

    double time_taken = timer.seconds();

    int64_t expected = N*(N+1)/2;
    if(result != expected)
      printf("result = %ld, expected = %ld\n",result, expected);

    return time_taken;
  }

  void run_test(int R) {
    double time_taken = 0.;
    time_taken = omp_for(R);
    printf("Time [omp-for] = %f\n",time_taken);

    time_taken = omp_for_kernel(R);
    printf("Time [omp-for-kernel] = %f\n",time_taken);

    time_taken = kk_for(R);
    printf("Time [kk-par-for] = %f\n",time_taken);

    printf("\n");
    time_taken = omp_reduce(R);
    printf("Time [omp-reduce] = %f\n",time_taken);

    time_taken = omp_reduce_kernel(R);
    printf("Time [omp-reduce-kernel] = %f\n",time_taken);

    time_taken = kk_reduce(R);
    printf("Time [kk-par-reduce] = %f\n",time_taken);
  }
};
