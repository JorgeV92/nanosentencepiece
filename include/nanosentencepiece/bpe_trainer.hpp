#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "nanosentencepiece/model.hpp"

namespace nanosentencepiece {

struct BpeTrainerOptions {
  std::size_t vocab_size = 128;
  std::size_t min_pair_frequency = 2;
  NormalizerOptions normalizer_options{};
  SpecialTokens special_tokens{};
};

class BpeTrainer {
 public:
  explicit BpeTrainer(BpeTrainerOptions options = {});

  Model TrainFromFiles(const std::vector<std::string>& paths) const;
  Model TrainFromLines(const std::vector<std::string>& lines) const;

  const BpeTrainerOptions& options() const noexcept;

 private:
  BpeTrainerOptions options_;
};

}  // namespace nanosentencepiece
