#include "nanosentencepiece/processor.hpp"

#include <stdexcept>
#include <utility>

#include "nanosentencepiece/utf8.hpp"
#include "unigram_utils.hpp"

namespace nanosentencepiece {

namespace detail {

struct ProcessorUnigramIndexCache {
  unigram::PieceIndex index;
};

}  // namespace detail

namespace {

std::vector<unigram::Piece> BuildUnigramPieces(const Model& model) {
  std::vector<unigram::Piece> pieces;
  pieces.reserve(model.vocabulary.Size());

  for (std::size_t id = 0; id < model.vocabulary.Pieces().size(); ++id) {
    const int piece_id = static_cast<int>(id);
    if (piece_id == model.vocabulary.unk_id() ||
        piece_id == model.vocabulary.bos_id() ||
        piece_id == model.vocabulary.eos_id() ||
        piece_id == model.vocabulary.pad_id()) {
      continue;
    }

    unigram::Piece piece;
    piece.piece = model.vocabulary.Pieces()[id];
    piece.symbols = SplitUtf8(piece.piece);
    piece.score = model.PieceScoreForId(piece_id);
    piece.external_index = id;
    pieces.push_back(std::move(piece));
  }

  return pieces;
}

std::shared_ptr<const detail::ProcessorUnigramIndexCache> BuildUnigramIndexCache(const Model& model) {
  if (!model.IsUnigram()) {
    return nullptr;
  }

  auto cache = std::make_shared<detail::ProcessorUnigramIndexCache>();
  cache->index = unigram::BuildPieceIndex(BuildUnigramPieces(model));
  return cache;
}

}  // namespace

SentencePieceProcessor::SentencePieceProcessor()
    : SentencePieceProcessor(std::make_shared<Model>()) {}

SentencePieceProcessor::SentencePieceProcessor(Model model)
    : SentencePieceProcessor(std::make_shared<Model>(std::move(model))) {}

SentencePieceProcessor::SentencePieceProcessor(std::shared_ptr<const Model> model)
    : model_(std::move(model)),
      unigram_index_cache_(model_ ? BuildUnigramIndexCache(*model_) : nullptr),
      normalizer_(model_ ? model_->normalizer_options : NormalizerOptions{}) {
  if (!model_) {
    throw std::invalid_argument("model pointer must not be null");
  }
}

SentencePieceProcessor::~SentencePieceProcessor() = default;

SentencePieceProcessor SentencePieceProcessor::Load(const std::string& path) {
  return SentencePieceProcessor(Model::Load(path));
}

std::vector<std::string> SentencePieceProcessor::ApplyMerges(
    const std::vector<std::string>& pieces) const {
  std::vector<std::string> current = pieces;

  for (const auto& merge : model_->merges) {
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

std::vector<std::string> SentencePieceProcessor::EncodeToPieces(
    std::string_view text,
    EncodeOptions options) const {
  std::vector<std::string> output;

  if (options.add_bos) {
    output.push_back(model_->special_tokens.bos);
  }

  const std::string escaped = normalizer_.NormalizeAndEscape(text);
  const std::vector<std::string> base_pieces = SplitUtf8(escaped);

  if (model_->IsUnigram()) {
    if (!unigram_index_cache_) {
      throw std::runtime_error("unigram model is missing cached index");
    }

    const auto& unigram_index = unigram_index_cache_->index;
    const auto path = unigram::BestPath(base_pieces, unigram_index, model_->special_tokens.unk);
    const auto pieces = unigram::MaterializePieces(path, unigram_index, model_->special_tokens.unk);
    output.insert(output.end(), pieces.begin(), pieces.end());
  } else {
    const std::vector<std::string> merged_pieces = ApplyMerges(base_pieces);

    for (const auto& piece : merged_pieces) {
      if (model_->vocabulary.Contains(piece)) {
        output.push_back(piece);
      } else {
        output.push_back(model_->special_tokens.unk);
      }
    }
  }

  if (options.add_eos) {
    output.push_back(model_->special_tokens.eos);
  }

  return output;
}

std::vector<int> SentencePieceProcessor::EncodeToIds(
    std::string_view text,
    EncodeOptions options) const {
  const auto pieces = EncodeToPieces(text, options);
  std::vector<int> ids;
  ids.reserve(pieces.size());
  for (const auto& piece : pieces) {
    ids.push_back(model_->vocabulary.IdForPiece(piece));
  }
  return ids;
}

std::string SentencePieceProcessor::DecodePieces(const std::vector<std::string>& pieces) const {
  std::vector<std::string> raw_pieces;
  raw_pieces.reserve(pieces.size());

  for (const auto& piece : pieces) {
    if (piece == model_->special_tokens.bos ||
        piece == model_->special_tokens.eos ||
        piece == model_->special_tokens.pad) {
      continue;
    }
    raw_pieces.push_back(piece);
  }

  return normalizer_.RestoreWhitespace(JoinPieces(raw_pieces));
}

std::string SentencePieceProcessor::DecodeIds(const std::vector<int>& ids) const {
  std::vector<std::string> pieces;
  pieces.reserve(ids.size());

  for (int id : ids) {
    pieces.push_back(model_->vocabulary.PieceForId(id));
  }

  return DecodePieces(pieces);
}

const Model& SentencePieceProcessor::model() const noexcept { return *model_; }

const std::shared_ptr<const Model>& SentencePieceProcessor::model_ptr() const noexcept {
  return model_;
}

}  // namespace nanosentencepiece
