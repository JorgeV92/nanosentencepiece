#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "nanosentencepiece/model.hpp"
#include "nanosentencepiece/normalization.hpp"

namespace nanosentencepiece {

class Tokenizer {
 public:
  explicit Tokenizer(Model model);

  std::vector<std::string> EncodeToPieces(
      std::string_view text,
      bool add_bos = false,
      bool add_eos = false) const;

  std::vector<int> EncodeToIds(
      std::string_view text,
      bool add_bos = false,
      bool add_eos = false) const;

  std::string DecodePieces(const std::vector<std::string>& pieces) const;
  std::string DecodeIds(const std::vector<int>& ids) const;

  const Model& model() const noexcept;

 private:
  std::vector<std::string> ApplyMerges(const std::vector<std::string>& pieces) const;

  Model model_;
  Normalizer normalizer_;
};

}  // namespace nanosentencepiece
