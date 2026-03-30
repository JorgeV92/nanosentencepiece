#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nanosentencepiece::cli {

struct ParsedArgs {
  std::unordered_map<std::string, std::string> values;
  std::unordered_set<std::string> flags;
  std::vector<std::string> positionals;

  bool HasFlag(std::string_view name) const;
  bool HasValue(std::string_view name) const;
  std::string Get(std::string_view name, const std::string& fallback = "") const;
};

ParsedArgs ParseArgs(int argc, char** argv);
int ParseInt(const std::string& value, const std::string& flag_name);
std::vector<int> ParseIds(std::string_view text);
std::vector<std::string> SplitOnAsciiWhitespace(std::string_view text);

}  // namespace nanosentencepiece::cli
