#include "nanosentencepiece/model.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nanosentencepiece {

namespace {

std::vector<std::string> SplitTab(std::string_view line) {
  std::vector<std::string> parts;
  std::size_t start = 0;
  while (start <= line.size()) {
    const std::size_t pos = line.find('\t', start);
    if (pos == std::string_view::npos) {
      parts.emplace_back(line.substr(start));
      break;
    }
    parts.emplace_back(line.substr(start, pos - start));
    start = pos + 1;
  }
  return parts;
}

bool ParseBool(const std::string& value) {
  if (value == "1" || value == "true") {
    return true;
  }
  if (value == "0" || value == "false") {
    return false;
  }
  throw std::runtime_error("failed to parse boolean value: " + value);
}

std::size_t ParseSizeT(const std::string& value, const std::string& what) {
  try {
    return static_cast<std::size_t>(std::stoull(value));
  } catch (const std::exception&) {
    throw std::runtime_error("failed to parse " + what + ": " + value);
  }
}

double ParseDouble(const std::string& value, const std::string& what) {
  try {
    return std::stod(value);
  } catch (const std::exception&) {
    throw std::runtime_error("failed to parse " + what + ": " + value);
  }
}

}  // namespace

std::string ToString(ModelType model_type) {
  switch (model_type) {
    case ModelType::kBpe:
      return "bpe";
    case ModelType::kUnigram:
      return "unigram";
  }

  throw std::invalid_argument("unknown model type");
}

ModelType ModelTypeFromString(std::string_view value) {
  if (value == "bpe") {
    return ModelType::kBpe;
  }
  if (value == "unigram") {
    return ModelType::kUnigram;
  }
  throw std::runtime_error("unknown model type: " + std::string(value));
}

void Model::Finalize() {
  merge_ranks_.clear();
  for (const auto& merge : merges) {
    merge_ranks_.emplace(merge.merged, merge.rank);
  }

  piece_scores.resize(vocabulary.Size(), 0.0);
}

std::size_t Model::MergeRank(const std::string& merged_piece) const {
  const auto it = merge_ranks_.find(merged_piece);
  if (it == merge_ranks_.end()) {
    throw std::out_of_range("merge piece not found: " + merged_piece);
  }
  return it->second;
}

bool Model::HasMerge(const std::string& merged_piece) const {
  return merge_ranks_.contains(merged_piece);
}

double Model::PieceScoreForId(int id) const {
  if (id < 0 || static_cast<std::size_t>(id) >= piece_scores.size()) {
    throw std::out_of_range("piece score id is out of range");
  }
  return piece_scores[static_cast<std::size_t>(id)];
}

double Model::PieceScore(std::string_view piece) const {
  return PieceScoreForId(vocabulary.IdForPiece(piece));
}

bool Model::IsUnigram() const noexcept {
  return metadata.model_type == ModelType::kUnigram;
}

void Model::Save(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open model file for writing: " + path);
  }

  out << "format\t" << metadata.format << "\n";
  out << "version\t" << metadata.version << "\n";
  out << "model_type\t" << ToString(metadata.model_type) << "\n";
  out << "trained_vocab_size\t" << metadata.trained_vocab_size << "\n";
  out << "normalizer.lowercase\t" << (normalizer_options.lowercase ? 1 : 0) << "\n";
  out << "normalizer.collapse_whitespace\t" << (normalizer_options.collapse_whitespace ? 1 : 0) << "\n";
  out << "normalizer.add_dummy_prefix\t" << (normalizer_options.add_dummy_prefix ? 1 : 0) << "\n";
  out << "normalizer.whitespace_symbol\t" << normalizer_options.whitespace_symbol << "\n";
  out << "special.unk\t" << special_tokens.unk << "\n";
  out << "special.bos\t" << special_tokens.bos << "\n";
  out << "special.eos\t" << special_tokens.eos << "\n";
  out << "special.pad\t" << special_tokens.pad << "\n";
  out << "special.unk_id\t" << vocabulary.unk_id() << "\n";
  out << "special.bos_id\t" << vocabulary.bos_id() << "\n";
  out << "special.eos_id\t" << vocabulary.eos_id() << "\n";
  out << "special.pad_id\t" << vocabulary.pad_id() << "\n";
  out << "vocab.size\t" << vocabulary.Size() << "\n";
  out << "merges.size\t" << merges.size() << "\n";

  for (std::size_t id = 0; id < vocabulary.Pieces().size(); ++id) {
    out << "piece\t" << id << "\t" << vocabulary.Pieces()[id] << "\n";
  }

  if (metadata.model_type == ModelType::kUnigram) {
    out << std::setprecision(17);
    for (std::size_t id = 0; id < piece_scores.size(); ++id) {
      out << "piece_score\t" << id << "\t" << piece_scores[id] << "\n";
    }
  }

  for (const auto& merge : merges) {
    out << "merge\t" << merge.rank << "\t" << merge.left << "\t" << merge.right << "\t" << merge.merged << "\n";
  }
}

Model Model::Load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open model file for reading: " + path);
  }

  Model model;
  Vocabulary vocab;
  int unk_id = -1;
  int bos_id = -1;
  int eos_id = -1;
  int pad_id = -1;

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }

    const auto parts = SplitTab(line);
    if (parts.empty()) {
      continue;
    }

    const std::string& key = parts[0];

    if (key == "format" && parts.size() >= 2) {
      model.metadata.format = parts[1];
    } else if (key == "version" && parts.size() >= 2) {
      model.metadata.version = parts[1];
    } else if (key == "model_type" && parts.size() >= 2) {
      model.metadata.model_type = ModelTypeFromString(parts[1]);
    } else if (key == "trained_vocab_size" && parts.size() >= 2) {
      model.metadata.trained_vocab_size = ParseSizeT(parts[1], "trained_vocab_size");
    } else if (key == "normalizer.lowercase" && parts.size() >= 2) {
      model.normalizer_options.lowercase = ParseBool(parts[1]);
    } else if (key == "normalizer.collapse_whitespace" && parts.size() >= 2) {
      model.normalizer_options.collapse_whitespace = ParseBool(parts[1]);
    } else if (key == "normalizer.add_dummy_prefix" && parts.size() >= 2) {
      model.normalizer_options.add_dummy_prefix = ParseBool(parts[1]);
    } else if (key == "normalizer.whitespace_symbol" && parts.size() >= 2) {
      model.normalizer_options.whitespace_symbol = parts[1];
    } else if (key == "special.unk" && parts.size() >= 2) {
      model.special_tokens.unk = parts[1];
    } else if (key == "special.bos" && parts.size() >= 2) {
      model.special_tokens.bos = parts[1];
    } else if (key == "special.eos" && parts.size() >= 2) {
      model.special_tokens.eos = parts[1];
    } else if (key == "special.pad" && parts.size() >= 2) {
      model.special_tokens.pad = parts[1];
    } else if (key == "special.unk_id" && parts.size() >= 2) {
      unk_id = static_cast<int>(ParseSizeT(parts[1], "special.unk_id"));
    } else if (key == "special.bos_id" && parts.size() >= 2) {
      bos_id = static_cast<int>(ParseSizeT(parts[1], "special.bos_id"));
    } else if (key == "special.eos_id" && parts.size() >= 2) {
      eos_id = static_cast<int>(ParseSizeT(parts[1], "special.eos_id"));
    } else if (key == "special.pad_id" && parts.size() >= 2) {
      pad_id = static_cast<int>(ParseSizeT(parts[1], "special.pad_id"));
    } else if (key == "piece" && parts.size() >= 3) {
      const std::size_t expected_id = ParseSizeT(parts[1], "piece id");
      const int actual_id = vocab.AddPiece(parts[2]);
      if (expected_id != static_cast<std::size_t>(actual_id)) {
        throw std::runtime_error("piece ids in model are not contiguous");
      }
      if (model.piece_scores.size() <= expected_id) {
        model.piece_scores.resize(expected_id + 1, 0.0);
      }
    } else if (key == "piece_score" && parts.size() >= 3) {
      const std::size_t id = ParseSizeT(parts[1], "piece_score id");
      if (model.piece_scores.size() <= id) {
        model.piece_scores.resize(id + 1, 0.0);
      }
      model.piece_scores[id] = ParseDouble(parts[2], "piece_score");
    } else if (key == "merge" && parts.size() >= 5) {
      MergeRule merge;
      merge.rank = ParseSizeT(parts[1], "merge rank");
      merge.left = parts[2];
      merge.right = parts[3];
      merge.merged = parts[4];
      model.merges.push_back(std::move(merge));
    }
  }

  vocab.SetSpecialIds(unk_id, bos_id, eos_id, pad_id);
  model.vocabulary = std::move(vocab);
  model.Finalize();
  return model;
}

}  // namespace nanosentencepiece
