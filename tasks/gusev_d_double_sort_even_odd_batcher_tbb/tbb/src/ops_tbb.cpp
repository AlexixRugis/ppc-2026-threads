#include "gusev_d_double_sort_even_odd_batcher_tbb/tbb/include/ops_tbb.hpp"

#include "gusev_d_double_sort_even_odd_batcher_tbb/common/include/common.hpp"

namespace gusev_d_double_sort_even_odd_batcher_tbb_task_threads {

DoubleSortEvenOddBatcherTBB::DoubleSortEvenOddBatcherTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool DoubleSortEvenOddBatcherTBB::ValidationImpl() {
  return GetOutput().empty();
}

bool DoubleSortEvenOddBatcherTBB::PreProcessingImpl() {
  input_data_ = GetInput();
  result_data_.clear();
  return true;
}

bool DoubleSortEvenOddBatcherTBB::RunImpl() {
  result_data_ = input_data_;
  return true;
}

bool DoubleSortEvenOddBatcherTBB::PostProcessingImpl() {
  GetOutput() = result_data_;
  return true;
}

}  // namespace gusev_d_double_sort_even_odd_batcher_tbb_task_threads
