#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "nanosentencepiece/cli.hpp"
#include "nanosentencepiece/model.hpp"
#include "nanosentencepiece/tokenizer.hpp"
#include "nanosentencepiece/version.hpp"

namespace nsp = nanosentencepiece;

namespace {

void PrintUsage() {
  std::cout
      << "nsp_decode " << nsp::kVersion << "\n"
      << "Usage:\n"
      << "  nsp_decode --model model.nsp --ids \"1 5 18 9\"\n"
      << "  nsp_decode --model model.nsp --pieces \"▁he llo ▁world\"\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = nsp::cli::ParseArgs(argc, argv);
    if (args.HasFlag("--help") || !args.HasValue("--model") ||
        (!args.HasValue("--ids") && !args.HasValue("--pieces"))) {
      PrintUsage();
      return args.HasFlag("--help") ? 0 : 1;
    }

    const nsp::Model model = nsp::Model::Load(args.Get("--model"));
    const nsp::Tokenizer tokenizer(model);

    if (args.HasValue("--ids")) {
      const auto ids = nsp::cli::ParseIds(args.Get("--ids"));
      std::cout << tokenizer.DecodeIds(ids) << "\n";
      return 0;
    }

    const auto pieces = nsp::cli::SplitOnAsciiWhitespace(args.Get("--pieces"));
    std::cout << tokenizer.DecodePieces(pieces) << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "nsp_decode error: " << ex.what() << "\n";
    return 1;
  }
}
