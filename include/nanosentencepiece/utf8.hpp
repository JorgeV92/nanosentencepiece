#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace nanosentencepiece {

std::vector<std::string> SplitUtf8(std::string_view text);
std::string JoinPieces(const std::vector<std::string>& pieces);

}  // namespace nanosentencepiece
