#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "gusev_d_double_sort_even_odd_batcher_tbb/common/include/common.hpp"
#include "gusev_d_double_sort_even_odd_batcher_tbb/tbb/include/ops_tbb.hpp"

namespace {

using gusev_d_double_sort_even_odd_batcher_tbb_task_threads::DoubleSortEvenOddBatcherTBB;
using gusev_d_double_sort_even_odd_batcher_tbb_task_threads::InType;
using gusev_d_double_sort_even_odd_batcher_tbb_task_threads::OutType;

class GusevDoubleSortEvenOddBatcherTbbEnabled : public ::testing::TestWithParam<int> {};

OutType RunTaskPipeline(const InType &input) {
  DoubleSortEvenOddBatcherTBB task(input);
  if (!task.Validation()) {
    throw std::runtime_error("Validation failed");
  }
  if (!task.PreProcessing()) {
    throw std::runtime_error("PreProcessing failed");
  }
  if (!task.Run()) {
    throw std::runtime_error("Run failed");
  }
  if (!task.PostProcessing()) {
    throw std::runtime_error("PostProcessing failed");
  }

  return task.GetOutput();
}

TEST_P(GusevDoubleSortEvenOddBatcherTbbEnabled, RunsSkeletonPipelineOnEmptyInput) {
  EXPECT_TRUE(RunTaskPipeline({}).empty());
}

std::string PrintTbbFunctionalParamName(const ::testing::TestParamInfo<int> &info) {
  static_cast<void>(info);
  return "enabled";
}

INSTANTIATE_TEST_SUITE_P(gusev_d_double_sort_even_odd_batcher_tbb_enabled, GusevDoubleSortEvenOddBatcherTbbEnabled,
                         ::testing::Values(0), PrintTbbFunctionalParamName);

}  // namespace
