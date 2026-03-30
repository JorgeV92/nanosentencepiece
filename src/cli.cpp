#include "nanosentencepiece/cli.hpp"

#include <cctype>
#include <sstream>
#include <stdexcept>

namespace nanosentencepiece::cli {

bool ParsedArgs::HasFlag(std::string_view name) const {
  return flags.contains(std::string(name));
}

bool ParsedArgs::HasValue(std::string_view name) const {
  return values.contains(std::string(name));
}

std::string ParsedArgs::Get(std::string_view name, const std::string& fallback) const {
  const auto it = values.find(std::string(name));
  if (it == values.end()) {
    return fallback;
  }
  return it->second;
}

ParsedArgs ParseArgs(int argc, char** argv) {
  ParsedArgs args;
  for (int i = 1; i < argc; ++i) {
    std::string current = argv[i];
    if (current.rfind("--", 0) == 0) {
      if (i + 1 < argc) {
        std::string next = argv[i + 1];
        if (next.rfind("--", 0) != 0) {
          args.values[current] = next;
          ++i;
          continue;
        }
      }
      args.flags.insert(current);
    } else {
      args.positionals.push_back(current);
    }
  }
  return args;
}

int ParseInt(const std::string& value, const std::string& flag_name) {
  try {
    return std::stoi(value);
  } catch (const std::exception&) {
    throw std::runtime_error("failed to parse integer for " + flag_name + ": " + value);
  }
}

std::vector<int> ParseIds(std::string_view text) {
  std::string canonical;
  canonical.reserve(text.size());
  for (char ch : text) {
    if (ch == ',') {
      canonical.push_back(' ');
    } else {
      canonical.push_back(ch);
    }
  }

  std::istringstream in(canonical);
  std::vector<int> ids;
  int value = 0;
  while (in >> value) {
    ids.push_back(value);
  }
  return ids;
}

std::vector<std::string> SplitOnAsciiWhitespace(std::string_view text) {
  std::vector<std::string> parts;
  std::string current;

  for (const unsigned char ch : text) {
    if (std::isspace(ch)) {
      if (!current.empty()) {
        parts.push_back(current);
        current.clear();
      }
    } else {
      current.push_back(static_cast<char>(ch));
    }
  }

  if (!current.empty()) {
    parts.push_back(current);
  }

  return parts;
}

}  // namespace nanosentencepiece::cli
