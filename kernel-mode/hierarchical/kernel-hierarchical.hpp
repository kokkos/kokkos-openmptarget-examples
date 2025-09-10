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

struct omp_kernel_hierarchical {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
  using team_policy = Kokkos::TeamPolicy<ExecSpace>;
  using member_type = team_policy::member_type;

  using view_1d = Kokkos::View<int64_t *, ExecSpace>;
  using view_2d = Kokkos::View<int64_t **, ExecSpace>;

  static const int nT = 32;
  int64_t N, M;
  view_1d vector;
  view_2d matrix;

  omp_kernel_hierarchical(int64_t N_)
      : N(N_), M(nT), matrix(view_2d("matrix",N,M)) {
  }

  double omp_hierarchical_for(int R)
  {
    int64_t N_ = N;
    int64_t M_ = M;

    // warmup
#pragma omp target teams distribute firstprivate(matrix)
  {
      for (int i = 0; i < N_; ++i) 
      {
#pragma omp parallel
        {
           #pragma omp for
          for (int j = 0; j < M_; ++j)
            matrix(i,j) = i+j;
        }
    }
  }
    
  Kokkos::Timer timer;
  for(int i = 0; i < R; ++i)
  {
#pragma omp target teams distribute firstprivate(matrix)
    {
      for (int i = 0; i < N_; ++i) 
      {
#pragma omp parallel
        {
          #pragma omp for
          for (int j = 0; j < M_; ++j)
            matrix(i,j) = i+j;
        }
      }
    }
  }

    double time_taken = timer.seconds();
    return time_taken;
  }

  double kk_hierarchical_for(int R)
  {
    int64_t N_ = N;
    int64_t M_ = M;

    // warmup
    Kokkos::parallel_for("kk-hierarchical-for", team_policy(N,nT), KOKKOS_LAMBDA(const member_type team){

      const int64_t i = team.league_rank();

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, M_), [=](const int j){
        matrix(i,j) = i+j;
      });
    });
    
  Kokkos::Timer timer;
  for(int i = 0; i < R; ++i)
  {
    Kokkos::parallel_for("kk-hierarchical-for", team_policy(N,nT), KOKKOS_LAMBDA(const member_type team){

      const int64_t i = team.league_rank();

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, M_), [=](const int j){
        matrix(i,j) = i+j;
      });
    });
  }

    double time_taken = timer.seconds();
    return time_taken;
  }

  double omp_hierarchical_for_kernel(int R)
  {
    int64_t N_ = N;
    int64_t M_ = M;

    const int nTeams = N_/nT + !! (N_%nT);

    // warmup
#pragma omp target teams ompx_bare num_teams(N_,1,1) thread_limit(nT,1,1) firstprivate(matrix)
    {
        const int blockIdx  = ompx::block_id(ompx::dim_x);
        const int blockDimx = ompx::block_dim(ompx::dim_x);
        const int threadIdx = ompx::thread_id(ompx::dim_x);
        const int index = blockIdx*blockDimx+threadIdx;

        for(int tid = threadIdx; tid < M_; tid+=blockDimx)
          matrix(blockIdx,tid) = blockIdx+tid;
    }
    
  Kokkos::Timer timer;
  for(int i = 0; i < R; ++i)
  {
#pragma omp target teams ompx_bare num_teams(N_,1,1) thread_limit(nT,1,1) firstprivate(matrix)
    {
        const int blockIdx  = ompx::block_id(ompx::dim_x);
        const int blockDimx = ompx::block_dim(ompx::dim_x);
        const int threadIdx = ompx::thread_id(ompx::dim_x);
        const int index = blockIdx*blockDimx+threadIdx;

        for(int tid = threadIdx; tid < M_; tid+=blockDimx)
          matrix(blockIdx,tid) = blockIdx+tid;
    }
  }

    double time_taken = timer.seconds();
    return time_taken;
  }

  double omp_reduce_hierarchical_kernel(int R)
  {
    int N_ = N;
    int M_ = M;
    int64_t result = 0;
    size_t scratch_size = nT*sizeof(int64_t);

  // FIXME: The number of teams generated should be dependent on other factors such as optimal number of threads per team, hardware limit on registers and other resources.
    const int nTeams = (N_ > 1024) ? 1024 : N_;

  {
    view_1d partial_results = view_1d("partial_results",nTeams);

#pragma omp target teams ompx_bare num_teams(nTeams,1,1) thread_limit(nT,1,1) ompx_dyn_cgroup_mem(scratch_size) firstprivate(matrix, partial_results)
      {
        const int blockIdx  = ompx::block_id(ompx::dim_x);
        const int blockDimx = ompx::block_dim(ompx::dim_x);
        const int threadIdx = ompx::thread_id(ompx::dim_x);
        const int gridDimx = ompx::grid_dim(ompx::dim_x);

        int64_t *buf = static_cast<int64_t*>(llvm_omp_target_dynamic_shared_alloc());
        buf[threadIdx] = 0;
        ompx_sync_block_acq_rel();

        for(int bid = blockIdx; bid < N_; bid+=gridDimx)
        {
          for(int tid = threadIdx; tid < M_; tid+=blockDimx)
            buf[threadIdx]+=matrix(bid,tid);

          ompx_sync_block_acq_rel();

          if(threadIdx == 0)
          {
            partial_results(blockIdx) = 0;

            for(int tid = 0; tid < blockDimx; ++tid)
              partial_results(blockIdx) += buf[tid];
          }
        }
      }
    }

    Kokkos::Timer timer;
    for(int i = 0; i < R; ++i)
    {
      view_1d partial_results = view_1d("partial_results",nTeams);
      result = 0;

#pragma omp target teams ompx_bare num_teams(nTeams,1,1) thread_limit(nT,1,1) ompx_dyn_cgroup_mem(scratch_size) firstprivate(matrix, partial_results)
      {
        const int blockIdx  = ompx::block_id(ompx::dim_x);
        const int blockDimx = ompx::block_dim(ompx::dim_x);
        const int threadIdx = ompx::thread_id(ompx::dim_x);
        const int gridDimx = ompx::grid_dim(ompx::dim_x);

        int64_t *buf = static_cast<int64_t*>(llvm_omp_target_dynamic_shared_alloc());
        buf[threadIdx] = 0;
        ompx_sync_block_acq_rel();

        for(int bid = blockIdx; bid < N_; bid+=gridDimx)
        {
          for(int tid = threadIdx; tid < M_; tid+=blockDimx)
            buf[threadIdx]+=matrix(bid,tid);

          ompx_sync_block_acq_rel();

          if(threadIdx == 0)
          {
            partial_results(blockIdx) = 0;
            for(int tid = 0; tid < blockDimx; ++tid)
              partial_results(blockIdx) += buf[tid];
          }
        }
      }

      auto h_partial_results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), partial_results);
      for(int i = 0; i < nTeams; ++i)
        result += h_partial_results(i);
    }

    double time_taken = timer.seconds();

    printf("result = %ld\n",result);
    return time_taken;
  }

  double kk_reduce_hierarchical(int R)
  {
    int N_ = N;
    int M_ = M;
    int64_t result = 0;

    Kokkos::parallel_reduce("kk-hierarchical-for", team_policy(N_,nT), KOKKOS_LAMBDA(const member_type team, int64_t& lsum){

      const int64_t i = team.league_rank();

      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, 0, N_), [=](const int j, int64_t& lsum_inner){
        lsum_inner += matrix(i,j);
      },lsum);
    },result);

/*#pragma omp target teams distribute firstprivate(matrix) reduction(+:result) thread_limit(nT)*/
/*    {*/
/*      for (int i = 0; i < N_; ++i) */
/*      {*/
/*        int64_t result_inner = 0;*/
/*#pragma omp parallel*/
/*        {*/
/*          #pragma omp for reduction(+:result_inner)*/
/*          for (int j = 0; j < N_; ++j)*/
/*            result_inner += matrix(i,j);*/
/**/
/*          result += result_inner;*/
/*        }*/
/*      }*/
/*    }*/

    Kokkos::Timer timer;
    /*for(int i = 0; i < R; ++i)*/
    /*{*/
    /*  result = 0;*/
    /*}*/

    double time_taken = timer.seconds();

    printf("result = %ld\n",result);
    return time_taken;
  }

  double omp_reduce_hierarchical(int R)
  {
    int N_ = N;
    int M_ = M;
    int64_t result = 0;
#pragma omp target teams distribute firstprivate(matrix) reduction(+:result) thread_limit(nT)
    {
      for (int i = 0; i < N_; ++i) 
      {
        int64_t result_inner = 0;
#pragma omp parallel
        {
          #pragma omp for reduction(+:result_inner)
          for (int j = 0; j < M_; ++j)
            result_inner += matrix(i,j);

          result += result_inner;
        }
      }
    }

    Kokkos::Timer timer;
    for(int i = 0; i < R; ++i)
    {
      result = 0;

#pragma omp target teams distribute firstprivate(matrix) reduction(+:result) thread_limit(nT)
      {
        for (int i = 0; i < N_; ++i) 
        {
          int64_t result_inner = 0;
#pragma omp parallel
          {
            #pragma omp for reduction(+:result_inner)
            for (int j = 0; j < M_; ++j)
              result_inner += matrix(i,j);

            result += result_inner;
          }
        }
      }
    }

    double time_taken = timer.seconds();

    printf("result = %ld\n",result);
    return time_taken;
  }

  void run_test(int R) {
    double time_taken = 0.;

    printf("\n Parallel-For \n");
    time_taken = omp_hierarchical_for(R);
    printf("Time [omp-hierarchical_for] = %f\n",time_taken);

    time_taken = omp_hierarchical_for_kernel(R);
    printf("Time [omp-hierarchical_for] = %f\n",time_taken);

    time_taken = kk_hierarchical_for(R);
    printf("Time [kk-hierarchical_for] = %f\n",time_taken);

    printf("\n Parallel-Reduce \n");
    time_taken = omp_reduce_hierarchical(R);
    printf("Time [omp-reduce-hierarchical] = %f\n",time_taken);

    time_taken = omp_reduce_hierarchical_kernel(R);
    printf("Time [omp-reduce-hierarchical-kernel] = %f\n",time_taken);

    time_taken = kk_reduce_hierarchical(R);
    printf("Time [kk-reduce-hierarchical] = %f\n",time_taken);
  }
};
