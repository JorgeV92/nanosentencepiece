#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "nanosentencepiece/model.hpp"

namespace nanosentencepiece {

struct UnigramTrainerOptions {
  std::size_t vocab_size = 128;
  std::size_t max_piece_length = 8;
  std::size_t min_piece_frequency = 2;
  std::size_t num_iterations = 4;
  std::size_t seed_piece_limit = 0;
  NormalizerOptions normalizer_options{};
  SpecialTokens special_tokens{};
};

class UnigramTrainer {
 public:
  explicit UnigramTrainer(UnigramTrainerOptions options = {});

  Model TrainFromFiles(const std::vector<std::string>& paths) const;
  Model TrainFromLines(const std::vector<std::string>& lines) const;

  const UnigramTrainerOptions& options() const noexcept;

 private:
  UnigramTrainerOptions options_;
};

}  // namespace nanosentencepiece
