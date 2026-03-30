#include "nanosentencepiece/normalization.hpp"

#include <algorithm>
#include <cctype>
#include <string>

namespace nanosentencepiece {

namespace {

std::string ToLowerAscii(std::string_view text) {
  std::string lowered;
  lowered.reserve(text.size());
  for (const unsigned char ch : text) {
    lowered.push_back(static_cast<char>(std::tolower(ch)));
  }
  return lowered;
}

std::string CollapseWhitespaceAscii(std::string_view text) {
  std::string out;
  out.reserve(text.size());

  bool in_whitespace = false;
  for (const unsigned char ch : text) {
    if (std::isspace(ch)) {
      if (!out.empty()) {
        in_whitespace = true;
      }
      continue;
    }

    if (in_whitespace && !out.empty()) {
      out.push_back(' ');
    }
    out.push_back(static_cast<char>(ch));
    in_whitespace = false;
  }

  return out;
}

void ReplaceAll(std::string& text, std::string_view from, std::string_view to) {
  if (from.empty()) {
    return;
  }

  std::size_t pos = 0;
  while ((pos = text.find(from.data(), pos, from.size())) != std::string::npos) {
    text.replace(pos, from.size(), to.data(), to.size());
    pos += to.size();
  }
}

}  // namespace

Normalizer::Normalizer(NormalizerOptions options) : options_(std::move(options)) {}

std::string Normalizer::Normalize(std::string_view text) const {
  std::string current(text);

  if (options_.lowercase) {
    current = ToLowerAscii(current);
  }

  if (options_.collapse_whitespace) {
    current = CollapseWhitespaceAscii(current);
  }

  return current;
}

std::string Normalizer::EscapeWhitespace(std::string_view normalized_text) const {
  std::string escaped;
  escaped.reserve(normalized_text.size() + options_.whitespace_symbol.size());

  if (normalized_text.empty()) {
    return escaped;
  }

  if (options_.add_dummy_prefix) {
    escaped += options_.whitespace_symbol;
  }

  for (char ch : normalized_text) {
    if (ch == ' ') {
      escaped += options_.whitespace_symbol;
    } else {
      escaped.push_back(ch);
    }
  }

  return escaped;
}

std::string Normalizer::RestoreWhitespace(std::string_view escaped_text) const {
  std::string restored(escaped_text);
  ReplaceAll(restored, options_.whitespace_symbol, " ");

  if (options_.add_dummy_prefix && !restored.empty() && restored.front() == ' ') {
    restored.erase(restored.begin());
  }

  return restored;
}

std::string Normalizer::NormalizeAndEscape(std::string_view text) const {
  return EscapeWhitespace(Normalize(text));
}

const NormalizerOptions& Normalizer::options() const noexcept { return options_; }

}  // namespace nanosentencepiece
