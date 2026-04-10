#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nanosentencepiece::unigram {

constexpr double kFallbackScore = -1.0e9;

struct Piece {
  std::string piece;
  std::vector<std::string> symbols;
  double score = 0.0;
  std::size_t external_index = 0;
};

struct PieceIndex {
  std::vector<Piece> pieces;
  std::unordered_map<std::string, std::vector<std::size_t>> by_first_symbol;
};

struct PathNode {
  bool is_fallback = false;
  std::size_t piece_index = 0;
};

inline PieceIndex BuildPieceIndex(std::vector<Piece> pieces) {
  PieceIndex index;
  index.pieces = std::move(pieces);

  for (std::size_t i = 0; i < index.pieces.size(); ++i) {
    const auto& piece = index.pieces[i];
    if (!piece.symbols.empty()) {
      index.by_first_symbol[piece.symbols.front()].push_back(i);
    }
  }

  for (auto& [first_symbol, ids] : index.by_first_symbol) {
    (void)first_symbol;
    std::sort(ids.begin(), ids.end(), [&](std::size_t left, std::size_t right) {
      const auto& left_piece = index.pieces[left];
      const auto& right_piece = index.pieces[right];
      if (left_piece.symbols.size() != right_piece.symbols.size()) {
        return left_piece.symbols.size() > right_piece.symbols.size();
      }
      if (left_piece.score != right_piece.score) {
        return left_piece.score > right_piece.score;
      }
      return left_piece.piece < right_piece.piece;
    });
  }

  return index;
}

inline bool MatchesAt(
    const std::vector<std::string>& input_symbols,
    std::size_t start,
    const Piece& piece) {
  if (start + piece.symbols.size() > input_symbols.size()) {
    return false;
  }

  for (std::size_t i = 0; i < piece.symbols.size(); ++i) {
    if (input_symbols[start + i] != piece.symbols[i]) {
      return false;
    }
  }

  return true;
}

inline std::vector<PathNode> BestPath(
    const std::vector<std::string>& input_symbols,
    const PieceIndex& index,
    std::string_view fallback_piece = {}) {
  struct Step {
    double score = -std::numeric_limits<double>::infinity();
    std::size_t prev = 0;
    std::size_t piece_index = 0;
    bool valid = false;
    bool is_fallback = false;
  };

  auto better_candidate = [&](double candidate_score, const Piece& candidate_piece, const Step& current) {
    if (!current.valid) {
      return true;
    }
    if (candidate_score != current.score) {
      return candidate_score > current.score;
    }
    if (current.is_fallback) {
      return true;
    }

    const auto& current_piece = index.pieces[current.piece_index];
    if (candidate_piece.symbols.size() != current_piece.symbols.size()) {
      return candidate_piece.symbols.size() > current_piece.symbols.size();
    }
    return candidate_piece.piece < current_piece.piece;
  };

  const auto better_fallback = [&](double candidate_score, const Step& current) {
    if (!current.valid) {
      return true;
    }
    return candidate_score > current.score;
  };

  std::vector<Step> best(input_symbols.size() + 1);
  best[0].score = 0.0;
  best[0].valid = true;

  for (std::size_t i = 0; i < input_symbols.size(); ++i) {
    if (!best[i].valid) {
      continue;
    }

    if (const auto it = index.by_first_symbol.find(input_symbols[i]); it != index.by_first_symbol.end()) {
      for (const std::size_t piece_index : it->second) {
        const auto& piece = index.pieces[piece_index];
        if (!MatchesAt(input_symbols, i, piece)) {
          continue;
        }

        const std::size_t end = i + piece.symbols.size();
        const double candidate_score = best[i].score + piece.score;
        if (better_candidate(candidate_score, piece, best[end])) {
          best[end].score = candidate_score;
          best[end].prev = i;
          best[end].piece_index = piece_index;
          best[end].valid = true;
          best[end].is_fallback = false;
        }
      }
    }

    if (!fallback_piece.empty()) {
      const std::size_t end = i + 1;
      const double candidate_score = best[i].score + kFallbackScore;
      if (better_fallback(candidate_score, best[end])) {
        best[end].score = candidate_score;
        best[end].prev = i;
        best[end].piece_index = 0;
        best[end].valid = true;
        best[end].is_fallback = true;
      }
    }
  }

  if (!best.back().valid) {
    throw std::runtime_error("failed to segment input with unigram model");
  }

  std::vector<PathNode> path;
  for (std::size_t pos = input_symbols.size(); pos > 0;) {
    const auto& step = best[pos];
    path.push_back(PathNode{step.is_fallback, step.piece_index});
    pos = step.prev;
  }
  std::reverse(path.begin(), path.end());
  return path;
}

inline std::vector<std::string> MaterializePieces(
    const std::vector<PathNode>& path,
    const PieceIndex& index,
    std::string_view fallback_piece) {
  std::vector<std::string> pieces;
  pieces.reserve(path.size());

  for (const auto& node : path) {
    if (node.is_fallback) {
      pieces.emplace_back(fallback_piece);
    } else {
      pieces.push_back(index.pieces[node.piece_index].piece);
    }
  }

  return pieces;
}

}  // namespace nanosentencepiece::unigram
