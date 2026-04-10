#include "nanosentencepiece/tokenizer.hpp"

namespace nanosentencepiece {

Tokenizer::Tokenizer(Model model) : processor_(std::move(model)) {}

std::vector<std::string> Tokenizer::EncodeToPieces(
    std::string_view text,
    bool add_bos,
    bool add_eos) const {
  return processor_.EncodeToPieces(text, EncodeOptions{add_bos, add_eos});
}

std::vector<int> Tokenizer::EncodeToIds(
    std::string_view text,
    bool add_bos,
    bool add_eos) const {
  return processor_.EncodeToIds(text, EncodeOptions{add_bos, add_eos});
}

std::string Tokenizer::DecodePieces(const std::vector<std::string>& pieces) const {
  return processor_.DecodePieces(pieces);
}

std::string Tokenizer::DecodeIds(const std::vector<int>& ids) const { return processor_.DecodeIds(ids); }

const Model& Tokenizer::model() const noexcept { return processor_.model(); }

}  // namespace nanosentencepiece
