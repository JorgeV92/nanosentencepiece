#include "nanosentencepiece/bpe_trainer.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "nanosentencepiece/normalization.hpp"
#include "nanosentencepiece/utf8.hpp"
#include "nanosentencepiece/vocabulary.hpp"

namespace nanosentencepiece {

namespace {

struct Sequence {
  std::vector<std::string> symbols;
};

struct PairKey {
  std::string left;
  std::string right;

  bool operator==(const PairKey& other) const {
    return left == other.left && right == other.right;
  }
};

struct PairKeyHash {
  std::size_t operator()(const PairKey& key) const {
    return std::hash<std::string>{}(key.left) ^ (std::hash<std::string>{}(key.right) << 1U);
  }
};

struct PairStats {
  PairKey pair;
  std::size_t frequency = 0;
};

std::vector<std::string> ReadAllLines(const std::vector<std::string>& paths) {
  std::vector<std::string> lines;
  for (const auto& path : paths) {
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("failed to open corpus file: " + path);
    }

    std::string line;
    while (std::getline(in, line)) {
      lines.push_back(line);
    }
  }
  return lines;
}

std::vector<Sequence> BuildSequences(
    const std::vector<std::string>& lines,
    const Normalizer& normalizer,
    std::unordered_set<std::string>* symbol_set) {
  std::vector<Sequence> sequences;
  sequences.reserve(lines.size());

  for (const auto& line : lines) {
    const std::string escaped = normalizer.NormalizeAndEscape(line);
    if (escaped.empty()) {
      continue;
    }

    Sequence sequence{SplitUtf8(escaped)};
    if (sequence.symbols.empty()) {
      continue;
    }

    for (const auto& symbol : sequence.symbols) {
      symbol_set->insert(symbol);
    }

    sequences.push_back(std::move(sequence));
  }

  return sequences;
}

std::unordered_map<PairKey, std::size_t, PairKeyHash> CountPairs(const std::vector<Sequence>& sequences) {
  std::unordered_map<PairKey, std::size_t, PairKeyHash> counts;
  for (const auto& sequence : sequences) {
    if (sequence.symbols.size() < 2) {
      continue;
    }

    for (std::size_t i = 0; i + 1 < sequence.symbols.size(); ++i) {
      PairKey key{sequence.symbols[i], sequence.symbols[i + 1]};
      counts[key] += 1;
    }
  }
  return counts;
}

bool BetterPair(const PairStats& candidate, const PairStats& incumbent) {
  if (candidate.frequency != incumbent.frequency) {
    return candidate.frequency > incumbent.frequency;
  }
  if (candidate.pair.left != incumbent.pair.left) {
    return candidate.pair.left < incumbent.pair.left;
  }
  return candidate.pair.right < incumbent.pair.right;
}

PairStats FindBestPair(
    const std::unordered_map<PairKey, std::size_t, PairKeyHash>& counts,
    std::size_t min_pair_frequency) {
  PairStats best;
  for (const auto& [pair, frequency] : counts) {
    if (frequency < min_pair_frequency) {
      continue;
    }

    PairStats candidate{pair, frequency};
    if (best.frequency == 0 || BetterPair(candidate, best)) {
      best = std::move(candidate);
    }
  }
  return best;
}

bool ApplyMerge(std::vector<Sequence>* sequences, const PairKey& pair, const std::string& merged) {
  bool changed = false;

  for (auto& sequence : *sequences) {
    if (sequence.symbols.size() < 2) {
      continue;
    }

    std::vector<std::string> next;
    next.reserve(sequence.symbols.size());

    for (std::size_t i = 0; i < sequence.symbols.size();) {
      if (i + 1 < sequence.symbols.size() &&
          sequence.symbols[i] == pair.left &&
          sequence.symbols[i + 1] == pair.right) {
        next.push_back(merged);
        i += 2;
        changed = true;
      } else {
        next.push_back(sequence.symbols[i]);
        ++i;
      }
    }

    sequence.symbols = std::move(next);
  }

  return changed;
}

std::vector<std::string> SortedSymbols(const std::unordered_set<std::string>& symbols) {
  std::vector<std::string> values(symbols.begin(), symbols.end());
  std::sort(values.begin(), values.end());
  return values;
}

}  // namespace

BpeTrainer::BpeTrainer(BpeTrainerOptions options) : options_(std::move(options)) {
  if (options_.vocab_size < 8) {
    throw std::invalid_argument("vocab_size must be at least large enough to hold special tokens and base pieces");
  }
}

Model BpeTrainer::TrainFromFiles(const std::vector<std::string>& paths) const {
  return TrainFromLines(ReadAllLines(paths));
}

Model BpeTrainer::TrainFromLines(const std::vector<std::string>& lines) const {
  Model model;
  model.normalizer_options = options_.normalizer_options;
  model.special_tokens = options_.special_tokens;
  model.metadata.trained_vocab_size = options_.vocab_size;

  const Normalizer normalizer(options_.normalizer_options);

  std::unordered_set<std::string> symbol_set;
  std::vector<Sequence> sequences = BuildSequences(lines, normalizer, &symbol_set);

  Vocabulary vocab = Vocabulary::WithSpecialTokens(options_.special_tokens);

  for (const auto& symbol : SortedSymbols(symbol_set)) {
    vocab.AddPiece(symbol);
  }

  std::size_t next_rank = 0;
  while (vocab.Size() < options_.vocab_size) {
    const auto pair_counts = CountPairs(sequences);
    const PairStats best = FindBestPair(pair_counts, options_.min_pair_frequency);
    if (best.frequency == 0) {
      break;
    }

    const std::string merged = best.pair.left + best.pair.right;
    if (!ApplyMerge(&sequences, best.pair, merged)) {
      break;
    }

    const int before = static_cast<int>(vocab.Size());
    vocab.AddPiece(merged);
    if (static_cast<int>(vocab.Size()) == before) {
      break;
    }

    MergeRule merge;
    merge.left = best.pair.left;
    merge.right = best.pair.right;
    merge.merged = merged;
    merge.rank = next_rank++;
    model.merges.push_back(std::move(merge));
  }

  model.vocabulary = std::move(vocab);
  model.Finalize();
  return model;
}

const BpeTrainerOptions& BpeTrainer::options() const noexcept { return options_; }

}  // namespace nanosentencepiece
