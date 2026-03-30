#include "nanosentencepiece/tokenizer.hpp"

#include <stdexcept>
#include <utility>

#include "nanosentencepiece/utf8.hpp"

namespace nanosentencepiece {

Tokenizer::Tokenizer(Model model)
    : model_(std::move(model)), normalizer_(model_.normalizer_options) {
  model_.Finalize();
}

std::vector<std::string> Tokenizer::ApplyMerges(const std::vector<std::string>& pieces) const {
  std::vector<std::string> current = pieces;

  for (const auto& merge : model_.merges) {
    if (current.size() < 2) {
      break;
    }

    std::vector<std::string> next;
    next.reserve(current.size());

    for (std::size_t i = 0; i < current.size();) {
      if (i + 1 < current.size() &&
          current[i] == merge.left &&
          current[i + 1] == merge.right) {
        next.push_back(merge.merged);
        i += 2;
      } else {
        next.push_back(current[i]);
        ++i;
      }
    }

    current = std::move(next);
  }

  return current;
}

std::vector<std::string> Tokenizer::EncodeToPieces(
    std::string_view text,
    bool add_bos,
    bool add_eos) const {
  std::vector<std::string> output;

  if (add_bos) {
    output.push_back(model_.special_tokens.bos);
  }

  const std::string escaped = normalizer_.NormalizeAndEscape(text);
  const std::vector<std::string> base_pieces = SplitUtf8(escaped);
  const std::vector<std::string> merged_pieces = ApplyMerges(base_pieces);

  for (const auto& piece : merged_pieces) {
    if (model_.vocabulary.Contains(piece)) {
      output.push_back(piece);
    } else {
      output.push_back(model_.special_tokens.unk);
    }
  }

  if (add_eos) {
    output.push_back(model_.special_tokens.eos);
  }

  return output;
}

std::vector<int> Tokenizer::EncodeToIds(
    std::string_view text,
    bool add_bos,
    bool add_eos) const {
  const auto pieces = EncodeToPieces(text, add_bos, add_eos);
  std::vector<int> ids;
  ids.reserve(pieces.size());
  for (const auto& piece : pieces) {
    ids.push_back(model_.vocabulary.IdForPiece(piece));
  }
  return ids;
}

std::string Tokenizer::DecodePieces(const std::vector<std::string>& pieces) const {
  std::vector<std::string> raw_pieces;
  raw_pieces.reserve(pieces.size());

  for (const auto& piece : pieces) {
    if (piece == model_.special_tokens.bos ||
        piece == model_.special_tokens.eos ||
        piece == model_.special_tokens.pad) {
      continue;
    }
    raw_pieces.push_back(piece);
  }

  return normalizer_.RestoreWhitespace(JoinPieces(raw_pieces));
}

std::string Tokenizer::DecodeIds(const std::vector<int>& ids) const {
  std::vector<std::string> pieces;
  pieces.reserve(ids.size());

  for (int id : ids) {
    pieces.push_back(model_.vocabulary.PieceForId(id));
  }

  return DecodePieces(pieces);
}

const Model& Tokenizer::model() const noexcept { return model_; }

}  // namespace nanosentencepiece
