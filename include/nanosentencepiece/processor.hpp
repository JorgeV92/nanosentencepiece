#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "nanosentencepiece/model.hpp"
#include "nanosentencepiece/normalization.hpp"

namespace nanosentencepiece {

namespace detail {
struct ProcessorUnigramIndexCache;
}

struct EncodeOptions {
  bool add_bos = false;
  bool add_eos = false;
};

class SentencePieceProcessor {
 public:
  SentencePieceProcessor();
  explicit SentencePieceProcessor(Model model);
  explicit SentencePieceProcessor(std::shared_ptr<const Model> model);
  ~SentencePieceProcessor();

  SentencePieceProcessor(const SentencePieceProcessor&) = default;
  SentencePieceProcessor(SentencePieceProcessor&&) noexcept = default;
  SentencePieceProcessor& operator=(const SentencePieceProcessor&) = default;
  SentencePieceProcessor& operator=(SentencePieceProcessor&&) noexcept = default;

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
  std::shared_ptr<const detail::ProcessorUnigramIndexCache> unigram_index_cache_;
  Normalizer normalizer_;
};

}  // namespace nanosentencepiece
