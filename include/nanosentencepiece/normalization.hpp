#pragma once

#include <string>
#include <string_view>

namespace nanosentencepiece {

struct NormalizerOptions {
  bool lowercase = true;
  bool collapse_whitespace = true;
  bool add_dummy_prefix = true;
  std::string whitespace_symbol = "▁";
};

class Normalizer {
 public:
  explicit Normalizer(NormalizerOptions options = NormalizerOptions{});

  std::string Normalize(std::string_view text) const;
  std::string EscapeWhitespace(std::string_view normalized_text) const;
  std::string RestoreWhitespace(std::string_view escaped_text) const;
  std::string NormalizeAndEscape(std::string_view text) const;

  const NormalizerOptions& options() const noexcept;

 private:
  NormalizerOptions options_;
};

}  // namespace nanosentencepiece
