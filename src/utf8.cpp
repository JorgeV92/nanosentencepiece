#include "nanosentencepiece/utf8.hpp"

#include <stdexcept>

namespace nanosentencepiece {

namespace {

std::size_t Utf8CharLength(unsigned char lead) {
  if ((lead & 0b1000'0000) == 0) {
    return 1;
  }
  if ((lead & 0b1110'0000) == 0b1100'0000) {
    return 2;
  }
  if ((lead & 0b1111'0000) == 0b1110'0000) {
    return 3;
  }
  if ((lead & 0b1111'1000) == 0b1111'0000) {
    return 4;
  }
  return 1;
}

bool IsContinuationByte(unsigned char byte) {
  return (byte & 0b1100'0000) == 0b1000'0000;
}

}  // namespace

std::vector<std::string> SplitUtf8(std::string_view text) {
  std::vector<std::string> pieces;
  pieces.reserve(text.size());

  std::size_t index = 0;
  while (index < text.size()) {
    const unsigned char lead = static_cast<unsigned char>(text[index]);
    const std::size_t length = Utf8CharLength(lead);

    if (length == 1 || index + length > text.size()) {
      pieces.emplace_back(text.substr(index, 1));
      ++index;
      continue;
    }

    bool valid = true;
    for (std::size_t offset = 1; offset < length; ++offset) {
      if (!IsContinuationByte(static_cast<unsigned char>(text[index + offset]))) {
        valid = false;
        break;
      }
    }

    if (!valid) {
      pieces.emplace_back(text.substr(index, 1));
      ++index;
      continue;
    }

    pieces.emplace_back(text.substr(index, length));
    index += length;
  }

  return pieces;
}

std::string JoinPieces(const std::vector<std::string>& pieces) {
  std::size_t total = 0;
  for (const auto& piece : pieces) {
    total += piece.size();
  }

  std::string joined;
  joined.reserve(total);
  for (const auto& piece : pieces) {
    joined += piece;
  }
  return joined;
}

}  // namespace nanosentencepiece
