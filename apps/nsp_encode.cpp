#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

#include "nanosentencepiece/cli.hpp"
#include "nanosentencepiece/model.hpp"
#include "nanosentencepiece/tokenizer.hpp"
#include "nanosentencepiece/version.hpp"

namespace nsp = nanosentencepiece;

namespace {

std::string ReadText(const nsp::cli::ParsedArgs& args) {
  if (args.HasValue("--text")) {
    return args.Get("--text");
  }
  if (args.HasValue("--input")) {
    std::ifstream in(args.Get("--input"), std::ios::binary);
    if (!in) {
      throw std::runtime_error("failed to open input text file: " + args.Get("--input"));
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  throw std::runtime_error("provide either --text or --input");
}

void PrintUsage() {
  std::cout
      << "nsp_encode " << nsp::kVersion << "\n"
      << "Usage:\n"
      << "  nsp_encode --model model.nsp --text \"hello world\" [--output pieces|ids] [--bos] [--eos]\n"
      << "  nsp_encode --model model.nsp --input text.txt [--output pieces|ids]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = nsp::cli::ParseArgs(argc, argv);
    if (args.HasFlag("--help") || !args.HasValue("--model")) {
      PrintUsage();
      return args.HasFlag("--help") ? 0 : 1;
    }

    const std::string text = ReadText(args);
    const nsp::Model model = nsp::Model::Load(args.Get("--model"));
    const nsp::Tokenizer tokenizer(model);

    const bool add_bos = args.HasFlag("--bos");
    const bool add_eos = args.HasFlag("--eos");
    const std::string mode = args.Get("--output", "pieces");

    if (mode == "pieces") {
      const auto pieces = tokenizer.EncodeToPieces(text, add_bos, add_eos);
      for (std::size_t i = 0; i < pieces.size(); ++i) {
        if (i > 0) {
          std::cout << ' ';
        }
        std::cout << pieces[i];
      }
      std::cout << "\n";
    } else if (mode == "ids") {
      const auto ids = tokenizer.EncodeToIds(text, add_bos, add_eos);
      for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) {
          std::cout << ' ';
        }
        std::cout << ids[i];
      }
      std::cout << "\n";
    } else {
      throw std::runtime_error("unknown --output mode: " + mode);
    }

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "nsp_encode error: " << ex.what() << "\n";
    return 1;
  }
}
