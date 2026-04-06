#include "baranov_a_mult_matrix_fox_algorithm/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "baranov_a_mult_matrix_fox_algorithm/common/include/common.hpp"

namespace baranov_a_mult_matrix_fox_algorithm_stl {

BaranovAMultMatrixFoxAlgorithmSTL::BaranovAMultMatrixFoxAlgorithmSTL(
    const baranov_a_mult_matrix_fox_algorithm::InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool BaranovAMultMatrixFoxAlgorithmSTL::ValidationImpl() {
  const auto& [matrix_size, matrix_a, matrix_b] = GetInput();
  return matrix_size > 0 && matrix_a.size() == matrix_size * matrix_size &&
         matrix_b.size() == matrix_size * matrix_size;
}

bool BaranovAMultMatrixFoxAlgorithmSTL::PreProcessingImpl() {
  const auto& [matrix_size, matrix_a, matrix_b] = GetInput();
  GetOutput() = std::vector<double>(matrix_size * matrix_size, 0.0);
  return true;
}

void BaranovAMultMatrixFoxAlgorithmSTL::StandardMultiplication(size_t n) {
  const auto& [matrix_size, matrix_a, matrix_b] = GetInput();
  auto& output = GetOutput();

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 4;
  
  std::vector<std::thread> threads;
  std::vector<size_t> ranges;

  size_t chunk_size = (n + num_threads - 1) / num_threads;
  for (unsigned int t = 0; t < num_threads; ++t) {
    size_t start_i = t * chunk_size;
    size_t end_i = std::min(start_i + chunk_size, n);
    if (start_i >= n) break;
    
    threads.emplace_back([&, start_i, end_i]() {
      for (size_t i = start_i; i < end_i; ++i) {
        for (size_t j = 0; j < n; ++j) {
          double sum = 0.0;
          for (size_t k = 0; k < n; ++k) {
            sum += matrix_a[(i * n) + k] * matrix_b[(k * n) + j];
          }
          output[(i * n) + j] = sum;
        }
      }
    });
  }
  
  for (auto& t : threads) {
    t.join();
  }
}

void BaranovAMultMatrixFoxAlgorithmSTL::FoxBlockMultiplication(size_t n, size_t block_size) {
  const auto& [matrix_size, matrix_a, matrix_b] = GetInput();
  auto& output = GetOutput();

  size_t num_blocks = (n + block_size - 1) / block_size;
  std::fill(output.begin(), output.end(), 0.0);

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 4;

  for (size_t bk = 0; bk < num_blocks; ++bk) {
    std::vector<std::thread> threads;
    std::vector<size_t> block_indices(num_blocks * num_blocks);
    for (size_t i = 0; i < num_blocks * num_blocks; ++i) {
      block_indices[i] = i;
    }

    size_t chunk_size = (block_indices.size() + num_threads - 1) / num_threads;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
      size_t start_idx = t * chunk_size;
      size_t end_idx = std::min(start_idx + chunk_size, block_indices.size());
      if (start_idx >= block_indices.size()) break;
      
      threads.emplace_back([&, start_idx, end_idx, bk]() {
        for (size_t idx = start_idx; idx < end_idx; ++idx) {
          size_t linear_idx = block_indices[idx];
          size_t bi = linear_idx / num_blocks;
          size_t bj = linear_idx % num_blocks;

          size_t broadcast_block = (bi + bk) % num_blocks;

          size_t i_start = bi * block_size;
          size_t i_end = std::min(i_start + block_size, n);
          size_t j_start = bj * block_size;
          size_t j_end = std::min(j_start + block_size, n);
          size_t k_start = broadcast_block * block_size;
          size_t k_end = std::min(k_start + block_size, n);

          for (size_t i = i_start; i < i_end; ++i) {
            for (size_t j = j_start; j < j_end; ++j) {
              double sum = 0.0;
              for (size_t k = k_start; k < k_end; ++k) {
                sum += matrix_a[(i * n) + k] * matrix_b[(k * n) + j];
              }
              output[(i * n) + j] += sum;
            }
          }
        }
      });
    }
    for (auto& t : threads) {
      t.join();
    }
  }
}

bool BaranovAMultMatrixFoxAlgorithmSTL::RunImpl() {
  const auto& [matrix_size, matrix_a, matrix_b] = GetInput();
  size_t n = matrix_size;
  size_t block_size = 64;
  if (n < block_size) {
    StandardMultiplication(n);
  } else {
    FoxBlockMultiplication(n, block_size);
  }

  return true;
}

bool BaranovAMultMatrixFoxAlgorithmSTL::PostProcessingImpl() {
  return true;
}

}  // namespace baranov_a_mult_matrix_fox_algorithm_stl
