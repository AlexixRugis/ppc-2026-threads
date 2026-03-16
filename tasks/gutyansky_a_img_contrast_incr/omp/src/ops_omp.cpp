#include "gutyansky_a_img_contrast_incr/omp/include/ops_omp.hpp"

#include <atomic>
#include <numeric>
#include <vector>

#include "gutyansky_a_img_contrast_incr/common/include/common.hpp"
#include "util/include/util.hpp"

namespace gutyansky_a_img_contrast_incr {

GutyanskyAImgContrastIncrOMP::GutyanskyAImgContrastIncrOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool GutyanskyAImgContrastIncrOMP::ValidationImpl() {
  return !GetInput().empty();
}

bool GutyanskyAImgContrastIncrOMP::PreProcessingImpl() {
  GetOutput().resize(GetInput().size());
  return true;
}

bool GutyanskyAImgContrastIncrOMP::RunImpl() {
  const size_t sz = GetInput().size();
  uint8_t lower_bound = std::numeric_limits<uint8_t>::max();
  uint8_t upper_bound = std::numeric_limits<uint8_t>::min();

#pragma omp parallel for reduction(min : lower_bound) reduction(max : upper_bound)
  for (size_t i = 0; i < sz; i++) {
    uint8_t val = GetInput()[i];
    if (val < lower_bound) {
      lower_bound = val;
    }
    if (val > upper_bound) {
      upper_bound = val;
    }
  }

  uint8_t delta = upper_bound - lower_bound;

  if (delta == 0) {
#pragma omp parallel for
    for (size_t i = 0; i < sz; i++) {
      GetOutput()[i] = GetInput()[i];
    }
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < sz; i++) {
      auto old_value = static_cast<uint16_t>(GetInput()[i]);
      uint16_t new_value = (std::numeric_limits<uint8_t>::max() * (old_value - lower_bound)) / delta;

      GetOutput()[i] = static_cast<uint8_t>(new_value);
    }
  }

  return true;
}

bool GutyanskyAImgContrastIncrOMP::PostProcessingImpl() {
  return true;
}

}  // namespace gutyansky_a_img_contrast_incr
