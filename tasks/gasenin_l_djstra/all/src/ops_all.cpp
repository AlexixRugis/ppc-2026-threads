#include "gasenin_l_djstra/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

#include "gasenin_l_djstra/common/include/common.hpp"

namespace gasenin_l_djstra {

GaseninLDjstraALL::GaseninLDjstraALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GaseninLDjstraALL::ValidationImpl() {
  return GetInput() > 0;
}

bool GaseninLDjstraALL::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  const InType n = GetInput();
  const InType inf = std::numeric_limits<InType>::max();

  const int chunk = n / size_;
  const int rem = n % size_;

  if (rank_ < rem) {
    local_n_ = chunk + 1;
    start_v_ = rank_ * local_n_;
  } else {
    local_n_ = chunk;
    start_v_ = rem * (chunk + 1) + (rank_ - rem) * chunk;
  }

  dist_.assign(local_n_, inf);
  visited_.assign(local_n_, 0);

  if (0 >= start_v_ && 0 < start_v_ + local_n_) {
    dist_[0 - start_v_] = 0;
  }

  return true;
}

static void MinPair(void *in, void *inout, int *len, MPI_Datatype * /*dtype*/) {
  InType *a = static_cast<InType *>(in);
  InType *b = static_cast<InType *>(inout);
  for (int i = 0; i < *len; i += 2) {
    if (a[i] < b[i]) {
      b[i] = a[i];
      b[i + 1] = a[i + 1];
    }
  }
}

bool GaseninLDjstraALL::RunImpl() {
  const InType n = GetInput();
  const InType inf = std::numeric_limits<InType>::max();

  std::vector<InType> &dist = dist_;
  std::vector<char> &visited = visited_;
  const int local_n = local_n_;
  const int start_v = start_v_;

  MPI_Op min_pair_op;
  MPI_Op_create(MinPair, 1, &min_pair_op);

  int num_threads = 1;
#pragma omp parallel
  {
#pragma omp single
    num_threads = omp_get_num_threads();
  }

  std::vector<InType> thread_mins(num_threads, inf);
  std::vector<InType> thread_vertices(num_threads, -1);

  for (int iteration = 0; iteration < n; ++iteration) {
#pragma omp parallel default(none) shared(local_n, start_v, dist, visited, thread_mins, thread_vertices, inf)
    {
      const int tid = omp_get_thread_num();
      InType t_min = inf;
      InType t_v = -1;

#pragma omp for nowait
      for (int i = 0; i < local_n; ++i) {
        if (visited[i] == 0 && dist[i] < t_min) {
          t_min = dist[i];
          t_v = start_v + i;
        }
      }

      thread_mins[tid] = t_min;
      thread_vertices[tid] = t_v;
    }

    InType local_min = inf;
    InType local_vertex = -1;
    for (int i = 0; i < num_threads; ++i) {
      if (thread_mins[i] < local_min) {
        local_min = thread_mins[i];
        local_vertex = thread_vertices[i];
      }
    }

    InType local_pair[2] = {local_min, local_vertex};
    InType global_pair[2] = {inf, -1};
    MPI_Allreduce(local_pair, global_pair, 1, MPI_LONG_LONG_INT, min_pair_op, MPI_COMM_WORLD);

    InType global_min = global_pair[0];
    InType global_vertex = global_pair[1];

    if (global_vertex == -1) {
      break;
    }

    if (global_vertex >= start_v && global_vertex < start_v + local_n) {
      visited[global_vertex - start_v] = 1;
    }

#pragma omp parallel for default(none) shared(local_n, start_v, dist, visited, global_vertex, global_min)
    for (int i = 0; i < local_n; ++i) {
      if (visited[i] == 0) {
        const InType global_i = start_v + i;
        if (global_i != global_vertex) {
          const InType weight = std::abs(global_vertex - global_i);
          const InType new_dist = global_min + weight;
          if (new_dist < dist[i]) {
            dist[i] = new_dist;
          }
        }
      }
    }
  }

  MPI_Op_free(&min_pair_op);

  int64_t local_sum = 0;
#pragma omp parallel for reduction(+ : local_sum) default(none) shared(local_n, dist, inf)
  for (int i = 0; i < local_n; ++i) {
    if (dist[i] != inf) {
      local_sum += dist[i];
    }
  }

  MPI_Allreduce(&local_sum, &total_sum_, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  return true;
}

bool GaseninLDjstraALL::PostProcessingImpl() {
  GetOutput() = static_cast<OutType>(total_sum_);
  return true;
}

}  // namespace gasenin_l_djstra
