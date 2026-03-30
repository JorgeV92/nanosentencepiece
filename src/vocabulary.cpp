#include "nanosentencepiece/vocabulary.hpp"

#include <stdexcept>

namespace nanosentencepiece {

Vocabulary Vocabulary::WithSpecialTokens(const SpecialTokens& special_tokens) {
  Vocabulary vocab;
  const int unk_id = vocab.AddPiece(special_tokens.unk);
  const int bos_id = vocab.AddPiece(special_tokens.bos);
  const int eos_id = vocab.AddPiece(special_tokens.eos);
  const int pad_id = vocab.AddPiece(special_tokens.pad);
  vocab.SetSpecialIds(unk_id, bos_id, eos_id, pad_id);
  return vocab;
}

int Vocabulary::AddPiece(const std::string& piece) {
  if (const auto it = piece_to_id_.find(piece); it != piece_to_id_.end()) {
    return it->second;
  }

  const int id = static_cast<int>(id_to_piece_.size());
  id_to_piece_.push_back(piece);
  piece_to_id_.emplace(piece, id);
  return id;
}

bool Vocabulary::Contains(std::string_view piece) const {
  return piece_to_id_.contains(std::string(piece));
}

int Vocabulary::IdForPiece(std::string_view piece) const {
  const auto it = piece_to_id_.find(std::string(piece));
  if (it == piece_to_id_.end()) {
    return unk_id_;
  }
  return it->second;
}

const std::string& Vocabulary::PieceForId(int id) const {
  if (id < 0 || static_cast<std::size_t>(id) >= id_to_piece_.size()) {
    throw std::out_of_range("piece id is out of range");
  }
  return id_to_piece_[static_cast<std::size_t>(id)];
}

std::size_t Vocabulary::Size() const noexcept { return id_to_piece_.size(); }

const std::vector<std::string>& Vocabulary::Pieces() const noexcept { return id_to_piece_; }

const std::unordered_map<std::string, int>& Vocabulary::PieceToId() const noexcept {
  return piece_to_id_;
}

int Vocabulary::unk_id() const noexcept { return unk_id_; }
int Vocabulary::bos_id() const noexcept { return bos_id_; }
int Vocabulary::eos_id() const noexcept { return eos_id_; }
int Vocabulary::pad_id() const noexcept { return pad_id_; }

void Vocabulary::SetSpecialIds(int unk_id, int bos_id, int eos_id, int pad_id) {
  unk_id_ = unk_id;
  bos_id_ = bos_id;
  eos_id_ = eos_id;
  pad_id_ = pad_id;
}

}  // namespace nanosentencepiece
