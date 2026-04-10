#include "nanosentencepiece/bpe_trainer.hpp"

#include <algorithm>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nanosentencepiece/normalization.hpp"
#include "nanosentencepiece/utf8.hpp"
#include "nanosentencepiece/vocabulary.hpp"

namespace nanosentencepiece {

namespace {

using SymbolId = std::uint32_t;

struct Sequence {
  std::vector<SymbolId> symbols;
};

class SymbolTable {
 public:
  SymbolId Intern(const std::string& symbol) {
    if (const auto it = symbol_to_id_.find(symbol); it != symbol_to_id_.end()) {
      return it->second;
    }

    const SymbolId id = static_cast<SymbolId>(symbols_.size());
    symbols_.push_back(symbol);
    symbol_to_id_.emplace(symbols_.back(), id);
    return id;
  }

  const std::string& Symbol(SymbolId id) const {
    return symbols_.at(static_cast<std::size_t>(id));
  }

  std::size_t Size() const noexcept { return symbols_.size(); }

 private:
  std::vector<std::string> symbols_;
  std::unordered_map<std::string, SymbolId> symbol_to_id_;
};

struct PairKey {
  SymbolId left = 0;
  SymbolId right = 0;

  bool operator==(const PairKey& other) const {
    return left == other.left && right == other.right;
  }
};

struct PairKeyHash {
  std::size_t operator()(const PairKey& key) const {
    return (static_cast<std::size_t>(key.left) * 1'000'003ULL) ^
           static_cast<std::size_t>(key.right);
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
    SymbolTable* symbol_table) {
  std::vector<Sequence> sequences;
  sequences.reserve(lines.size());

  for (const auto& line : lines) {
    const std::string escaped = normalizer.NormalizeAndEscape(line);
    if (escaped.empty()) {
      continue;
    }

    const std::vector<std::string> pieces = SplitUtf8(escaped);
    if (pieces.empty()) {
      continue;
    }

    Sequence sequence;
    sequence.symbols.reserve(pieces.size());
    for (const auto& symbol : pieces) {
      sequence.symbols.push_back(symbol_table->Intern(symbol));
    }

    sequences.push_back(std::move(sequence));
  }

  return sequences;
}

std::size_t CountAdjacentPairs(const std::vector<Sequence>& sequences) {
  std::size_t total = 0;
  for (const auto& sequence : sequences) {
    if (sequence.symbols.size() >= 2) {
      total += sequence.symbols.size() - 1;
    }
  }
  return total;
}

std::unordered_map<PairKey, std::size_t, PairKeyHash> CountPairs(const std::vector<Sequence>& sequences) {
  std::unordered_map<PairKey, std::size_t, PairKeyHash> counts;
  counts.reserve(CountAdjacentPairs(sequences));

  for (const auto& sequence : sequences) {
    if (sequence.symbols.size() < 2) {
      continue;
    }

    for (std::size_t i = 0; i + 1 < sequence.symbols.size(); ++i) {
      const PairKey key{sequence.symbols[i], sequence.symbols[i + 1]};
      counts[key] += 1;
    }
  }
  return counts;
}

bool BetterPair(
    const PairStats& candidate,
    const PairStats& incumbent,
    const SymbolTable& symbol_table) {
  if (candidate.frequency != incumbent.frequency) {
    return candidate.frequency > incumbent.frequency;
  }

  const std::string& candidate_left = symbol_table.Symbol(candidate.pair.left);
  const std::string& incumbent_left = symbol_table.Symbol(incumbent.pair.left);
  if (candidate_left != incumbent_left) {
    return candidate_left < incumbent_left;
  }

  return symbol_table.Symbol(candidate.pair.right) < symbol_table.Symbol(incumbent.pair.right);
}

PairStats FindBestPair(
    const std::unordered_map<PairKey, std::size_t, PairKeyHash>& counts,
    const SymbolTable& symbol_table,
    std::size_t min_pair_frequency) {
  PairStats best;
  for (const auto& [pair, frequency] : counts) {
    if (frequency < min_pair_frequency) {
      continue;
    }

    PairStats candidate{pair, frequency};
    if (best.frequency == 0 || BetterPair(candidate, best, symbol_table)) {
      best = std::move(candidate);
    }
  }
  return best;
}

bool ApplyMerge(std::vector<Sequence>* sequences, const PairKey& pair, SymbolId merged_id) {
  bool changed = false;

  for (auto& sequence : *sequences) {
    if (sequence.symbols.size() < 2) {
      continue;
    }

    std::vector<SymbolId> next;
    next.reserve(sequence.symbols.size());

    for (std::size_t i = 0; i < sequence.symbols.size();) {
      if (i + 1 < sequence.symbols.size() &&
          sequence.symbols[i] == pair.left &&
          sequence.symbols[i + 1] == pair.right) {
        next.push_back(merged_id);
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

std::vector<std::string> SortedBaseSymbols(
    const SymbolTable& symbol_table,
    std::size_t base_symbol_count) {
  std::vector<std::string> values;
  values.reserve(base_symbol_count);
  for (std::size_t i = 0; i < base_symbol_count; ++i) {
    values.push_back(symbol_table.Symbol(static_cast<SymbolId>(i)));
  }
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

  SymbolTable symbol_table;
  std::vector<Sequence> sequences = BuildSequences(lines, normalizer, &symbol_table);
  const std::size_t base_symbol_count = symbol_table.Size();

  Vocabulary vocab = Vocabulary::WithSpecialTokens(options_.special_tokens);

  for (const auto& symbol : SortedBaseSymbols(symbol_table, base_symbol_count)) {
    vocab.AddPiece(symbol);
  }

  std::size_t next_rank = 0;
  while (vocab.Size() < options_.vocab_size) {
    const auto pair_counts = CountPairs(sequences);
    const PairStats best = FindBestPair(pair_counts, symbol_table, options_.min_pair_frequency);
    if (best.frequency == 0) {
      break;
    }

    const std::string merged =
        symbol_table.Symbol(best.pair.left) + symbol_table.Symbol(best.pair.right);
    const SymbolId merged_id = symbol_table.Intern(merged);
    if (!ApplyMerge(&sequences, best.pair, merged_id)) {
      break;
    }

    const int before = static_cast<int>(vocab.Size());
    vocab.AddPiece(merged);
    if (static_cast<int>(vocab.Size()) == before) {
      break;
    }

    MergeRule merge;
    merge.left = symbol_table.Symbol(best.pair.left);
    merge.right = symbol_table.Symbol(best.pair.right);
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
