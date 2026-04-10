#include "nanosentencepiece/unigram_trainer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "nanosentencepiece/normalization.hpp"
#include "nanosentencepiece/utf8.hpp"
#include "nanosentencepiece/vocabulary.hpp"
#include "unigram_utils.hpp"

namespace nanosentencepiece {

namespace {

struct Sequence {
  std::vector<std::string> symbols;
};

struct CandidatePiece {
  unigram::Piece piece;
  std::size_t seed_frequency = 0;
  std::size_t used_count = 0;
  bool required = false;
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
    const Normalizer& normalizer) {
  std::vector<Sequence> sequences;
  sequences.reserve(lines.size());

  for (const auto& line : lines) {
    const std::string escaped = normalizer.NormalizeAndEscape(line);
    if (escaped.empty()) {
      continue;
    }

    const auto symbols = SplitUtf8(escaped);
    if (symbols.empty()) {
      continue;
    }

    sequences.push_back(Sequence{symbols});
  }

  return sequences;
}

std::unordered_map<std::string, std::size_t> CountCandidatePieces(
    const std::vector<Sequence>& sequences,
    std::size_t max_piece_length) {
  std::unordered_map<std::string, std::size_t> counts;

  for (const auto& sequence : sequences) {
    for (std::size_t start = 0; start < sequence.symbols.size(); ++start) {
      std::string piece;
      for (std::size_t length = 1;
           length <= max_piece_length && start + length <= sequence.symbols.size();
           ++length) {
        piece += sequence.symbols[start + length - 1];
        counts[piece] += 1;
      }
    }
  }

  return counts;
}

bool BetterSeedCandidate(const CandidatePiece& left, const CandidatePiece& right) {
  if (left.seed_frequency != right.seed_frequency) {
    return left.seed_frequency > right.seed_frequency;
  }
  if (left.piece.symbols.size() != right.piece.symbols.size()) {
    return left.piece.symbols.size() > right.piece.symbols.size();
  }
  return left.piece.piece < right.piece.piece;
}

std::vector<CandidatePiece> BuildSeedVocabulary(
    const std::unordered_map<std::string, std::size_t>& counts,
    const UnigramTrainerOptions& options,
    std::size_t regular_target_size) {
  std::vector<CandidatePiece> singles;
  std::vector<CandidatePiece> multi_pieces;

  for (const auto& [piece, frequency] : counts) {
    const auto symbols = SplitUtf8(piece);
    if (symbols.empty()) {
      continue;
    }
    if (symbols.size() > 1 && frequency < options.min_piece_frequency) {
      continue;
    }

    CandidatePiece candidate;
    candidate.piece.piece = piece;
    candidate.piece.symbols = symbols;
    candidate.seed_frequency = frequency;
    candidate.required = symbols.size() == 1;

    if (candidate.required) {
      singles.push_back(std::move(candidate));
    } else {
      multi_pieces.push_back(std::move(candidate));
    }
  }

  std::sort(singles.begin(), singles.end(), BetterSeedCandidate);
  std::sort(multi_pieces.begin(), multi_pieces.end(), BetterSeedCandidate);

  std::vector<CandidatePiece> selected = singles;
  const std::size_t default_seed_limit =
      std::max(regular_target_size * 4, selected.size());
  const std::size_t seed_limit =
      options.seed_piece_limit == 0 ? default_seed_limit : std::max(options.seed_piece_limit, selected.size());

  for (const auto& candidate : multi_pieces) {
    if (selected.size() >= seed_limit) {
      break;
    }
    selected.push_back(candidate);
  }

  double total_seed_frequency = 0.0;
  for (const auto& candidate : selected) {
    total_seed_frequency += static_cast<double>(candidate.seed_frequency);
  }
  if (total_seed_frequency == 0.0) {
    total_seed_frequency = 1.0;
  }

  for (std::size_t i = 0; i < selected.size(); ++i) {
    selected[i].piece.score =
        std::log(static_cast<double>(selected[i].seed_frequency) / total_seed_frequency);
    selected[i].piece.external_index = i;
  }

  return selected;
}

void RunHardEm(
    const std::vector<Sequence>& sequences,
    std::size_t num_iterations,
    std::vector<CandidatePiece>* candidates) {
  for (std::size_t iteration = 0; iteration < num_iterations; ++iteration) {
    std::vector<unigram::Piece> pieces;
    pieces.reserve(candidates->size());
    for (const auto& candidate : *candidates) {
      pieces.push_back(candidate.piece);
    }

    const auto index = unigram::BuildPieceIndex(std::move(pieces));
    std::vector<std::size_t> counts(candidates->size(), 0);
    std::size_t total_count = 0;

    for (const auto& sequence : sequences) {
      const auto path = unigram::BestPath(sequence.symbols, index);
      for (const auto& node : path) {
        if (node.is_fallback) {
          continue;
        }

        const auto candidate_index = index.pieces[node.piece_index].external_index;
        counts[candidate_index] += 1;
        total_count += 1;
      }
    }

    if (total_count == 0) {
      break;
    }

    for (std::size_t i = 0; i < candidates->size(); ++i) {
      (*candidates)[i].used_count = counts[i];
      if (counts[i] == 0 && !(*candidates)[i].required) {
        (*candidates)[i].piece.score = unigram::kFallbackScore;
      } else {
        const double normalized_count =
            static_cast<double>(std::max<std::size_t>(counts[i], 1));
        (*candidates)[i].piece.score = std::log(normalized_count / static_cast<double>(total_count));
      }
    }
  }
}

bool BetterFinalCandidate(const CandidatePiece& left, const CandidatePiece& right) {
  if (left.required != right.required) {
    return left.required && !right.required;
  }
  if (left.piece.score != right.piece.score) {
    return left.piece.score > right.piece.score;
  }
  if (left.used_count != right.used_count) {
    return left.used_count > right.used_count;
  }
  if (left.piece.symbols.size() != right.piece.symbols.size()) {
    return left.piece.symbols.size() > right.piece.symbols.size();
  }
  return left.piece.piece < right.piece.piece;
}

std::vector<CandidatePiece> SelectFinalVocabulary(
    std::vector<CandidatePiece> candidates,
    std::size_t regular_target_size) {
  std::sort(candidates.begin(), candidates.end(), BetterFinalCandidate);

  std::vector<CandidatePiece> selected;
  std::unordered_set<std::string> seen;

  for (const auto& candidate : candidates) {
    if (candidate.required) {
      if (seen.insert(candidate.piece.piece).second) {
        selected.push_back(candidate);
      }
    }
  }

  for (const auto& candidate : candidates) {
    if (selected.size() >= regular_target_size) {
      break;
    }
    if (seen.insert(candidate.piece.piece).second) {
      selected.push_back(candidate);
    }
  }

  std::sort(selected.begin(), selected.end(), [](const CandidatePiece& left, const CandidatePiece& right) {
    if (left.piece.score != right.piece.score) {
      return left.piece.score > right.piece.score;
    }
    if (left.piece.symbols.size() != right.piece.symbols.size()) {
      return left.piece.symbols.size() > right.piece.symbols.size();
    }
    return left.piece.piece < right.piece.piece;
  });

  return selected;
}

}  // namespace

UnigramTrainer::UnigramTrainer(UnigramTrainerOptions options)
    : options_(std::move(options)) {
  if (options_.vocab_size < 8) {
    throw std::invalid_argument("vocab_size must be at least large enough to hold special tokens and base pieces");
  }
  if (options_.max_piece_length == 0) {
    throw std::invalid_argument("max_piece_length must be greater than zero");
  }
  if (options_.num_iterations == 0) {
    throw std::invalid_argument("num_iterations must be greater than zero");
  }
}

Model UnigramTrainer::TrainFromFiles(const std::vector<std::string>& paths) const {
  return TrainFromLines(ReadAllLines(paths));
}

Model UnigramTrainer::TrainFromLines(const std::vector<std::string>& lines) const {
  Model model;
  model.metadata.model_type = ModelType::kUnigram;
  model.metadata.trained_vocab_size = options_.vocab_size;
  model.normalizer_options = options_.normalizer_options;
  model.special_tokens = options_.special_tokens;

  const Normalizer normalizer(options_.normalizer_options);
  const auto sequences = BuildSequences(lines, normalizer);
  const auto counts = CountCandidatePieces(sequences, options_.max_piece_length);

  Vocabulary vocab = Vocabulary::WithSpecialTokens(options_.special_tokens);
  const std::size_t special_piece_count = vocab.Size();
  const std::size_t regular_target_size =
      options_.vocab_size > special_piece_count ? options_.vocab_size - special_piece_count : 0;

  auto candidates = BuildSeedVocabulary(counts, options_, regular_target_size);
  RunHardEm(sequences, options_.num_iterations, &candidates);
  const auto final_pieces = SelectFinalVocabulary(std::move(candidates), regular_target_size);

  for (const auto& candidate : final_pieces) {
    vocab.AddPiece(candidate.piece.piece);
  }

  model.vocabulary = std::move(vocab);
  model.piece_scores.resize(model.vocabulary.Size(), 0.0);
  for (const auto& candidate : final_pieces) {
    model.piece_scores[static_cast<std::size_t>(model.vocabulary.IdForPiece(candidate.piece.piece))] =
        candidate.piece.score;
  }

  model.Finalize();
  return model;
}

const UnigramTrainerOptions& UnigramTrainer::options() const noexcept { return options_; }

}  // namespace nanosentencepiece
