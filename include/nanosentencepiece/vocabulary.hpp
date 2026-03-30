#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nanosentencepiece {

struct SpecialTokens {
  std::string unk = "<unk>";
  std::string bos = "<bos>";
  std::string eos = "<eos>";
  std::string pad = "<pad>";
};

class Vocabulary {
 public:
  Vocabulary() = default;

  static Vocabulary WithSpecialTokens(const SpecialTokens& special_tokens);

  int AddPiece(const std::string& piece);
  bool Contains(std::string_view piece) const;
  int IdForPiece(std::string_view piece) const;
  const std::string& PieceForId(int id) const;
  std::size_t Size() const noexcept;

  const std::vector<std::string>& Pieces() const noexcept;
  const std::unordered_map<std::string, int>& PieceToId() const noexcept;

  int unk_id() const noexcept;
  int bos_id() const noexcept;
  int eos_id() const noexcept;
  int pad_id() const noexcept;

  void SetSpecialIds(int unk_id, int bos_id, int eos_id, int pad_id);

 private:
  std::vector<std::string> id_to_piece_;
  std::unordered_map<std::string, int> piece_to_id_;
  int unk_id_ = -1;
  int bos_id_ = -1;
  int eos_id_ = -1;
  int pad_id_ = -1;
};

}  // namespace nanosentencepiece
