#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "nanosentencepiece/normalization.hpp"
#include "nanosentencepiece/vocabulary.hpp"

namespace nanosentencepiece {

struct MergeRule {
  std::string left;
  std::string right;
  std::string merged;
  std::size_t rank = 0;
};

struct ModelMetadata {
  std::string format = "NSPM";
  std::string version = "1";
  std::size_t trained_vocab_size = 0;
};

class Model {
 public:
  Model() = default;

  void Finalize();

  void Save(const std::string& path) const;
  static Model Load(const std::string& path);

  std::size_t MergeRank(const std::string& merged_piece) const;
  bool HasMerge(const std::string& merged_piece) const;

  NormalizerOptions normalizer_options;
  SpecialTokens special_tokens;
  Vocabulary vocabulary;
  std::vector<MergeRule> merges;
  ModelMetadata metadata;

 private:
  std::unordered_map<std::string, std::size_t> merge_ranks_;
};

}  // namespace nanosentencepiece
