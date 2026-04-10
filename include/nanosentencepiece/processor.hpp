#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "nanosentencepiece/model.hpp"
#include "nanosentencepiece/normalization.hpp"

namespace nanosentencepiece {

struct EncodeOptions {
  bool add_bos = false;
  bool add_eos = false;
};

class SentencePieceProcessor {
 public:
  SentencePieceProcessor();
  explicit SentencePieceProcessor(Model model);
  explicit SentencePieceProcessor(std::shared_ptr<const Model> model);

  static SentencePieceProcessor Load(const std::string& path);

  std::vector<std::string> EncodeToPieces(
      std::string_view text,
      EncodeOptions options = {}) const;

  std::vector<int> EncodeToIds(
      std::string_view text,
      EncodeOptions options = {}) const;

  std::string DecodePieces(const std::vector<std::string>& pieces) const;
  std::string DecodeIds(const std::vector<int>& ids) const;

  const Model& model() const noexcept;
  const std::shared_ptr<const Model>& model_ptr() const noexcept;

 private:
  std::vector<std::string> ApplyMerges(const std::vector<std::string>& pieces) const;

  std::shared_ptr<const Model> model_;
  Normalizer normalizer_;
};

}  // namespace nanosentencepiece
