#include <algorithm>
#include <exception>
#include <iostream>
#include <string>

#include "nanosentencepiece/cli.hpp"
#include "nanosentencepiece/model.hpp"
#include "nanosentencepiece/version.hpp"

namespace nsp = nanosentencepiece;

namespace {

void PrintUsage() {
  std::cout
      << "nsp_inspect " << nsp::kVersion << "\n"
      << "Usage:\n"
      << "  nsp_inspect --model model.nsp [--limit 32]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = nsp::cli::ParseArgs(argc, argv);
    if (args.HasFlag("--help") || !args.HasValue("--model")) {
      PrintUsage();
      return args.HasFlag("--help") ? 0 : 1;
    }

    const nsp::Model model = nsp::Model::Load(args.Get("--model"));
    const std::size_t limit = args.HasValue("--limit")
                                  ? static_cast<std::size_t>(nsp::cli::ParseInt(args.Get("--limit"), "--limit"))
                                  : 32;

    std::cout << "model format: " << model.metadata.format << " v" << model.metadata.version << "\n";
    std::cout << "model type: " << nsp::ToString(model.metadata.model_type) << "\n";
    std::cout << "trained vocab size target: " << model.metadata.trained_vocab_size << "\n";
    std::cout << "actual vocab size: " << model.vocabulary.Size() << "\n";
    std::cout << "merge count: " << model.merges.size() << "\n";
    std::cout << "normalizer.lowercase: " << model.normalizer_options.lowercase << "\n";
    std::cout << "normalizer.collapse_whitespace: " << model.normalizer_options.collapse_whitespace << "\n";
    std::cout << "normalizer.add_dummy_prefix: " << model.normalizer_options.add_dummy_prefix << "\n";
    std::cout << "whitespace symbol: " << model.normalizer_options.whitespace_symbol << "\n\n";

    std::cout << "vocabulary preview:\n";
    const auto& pieces = model.vocabulary.Pieces();
    for (std::size_t i = 0; i < std::min(limit, pieces.size()); ++i) {
      std::cout << "  [" << i << "] " << pieces[i] << "\n";
    }

    if (model.IsUnigram()) {
      std::cout << "\nunigram piece scores:\n";
      for (std::size_t i = 0; i < std::min(limit, pieces.size()); ++i) {
        std::cout << "  [" << i << "] " << pieces[i]
                  << " score=" << model.PieceScoreForId(static_cast<int>(i)) << "\n";
      }
    } else {
      std::cout << "\nmerge preview:\n";
      for (std::size_t i = 0; i < std::min(limit, model.merges.size()); ++i) {
        const auto& merge = model.merges[i];
        std::cout << "  #" << merge.rank << ": (" << merge.left << ", " << merge.right
                  << ") -> " << merge.merged << "\n";
      }
    }

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "nsp_inspect error: " << ex.what() << "\n";
    return 1;
  }
}
