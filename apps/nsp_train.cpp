#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "nanosentencepiece/bpe_trainer.hpp"
#include "nanosentencepiece/cli.hpp"
#include "nanosentencepiece/model.hpp"
#include "nanosentencepiece/version.hpp"

namespace nsp = nanosentencepiece;

namespace {

void PrintUsage() {
  std::cout
      << "nsp_train " << nsp::kVersion << "\n"
      << "Usage:\n"
      << "  nsp_train --model model.nsp --vocab-size 128 [--min-pair-frequency 2]\n"
      << "            [--no-lowercase] [--no-collapse-whitespace]\n"
      << "            corpus1.txt [corpus2.txt ...]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = nsp::cli::ParseArgs(argc, argv);

    if (args.HasFlag("--help") || args.positionals.empty() || !args.HasValue("--model")) {
      PrintUsage();
      return args.HasFlag("--help") ? 0 : 1;
    }

    nsp::BpeTrainerOptions options;
    if (args.HasValue("--vocab-size")) {
      options.vocab_size = static_cast<std::size_t>(
          nsp::cli::ParseInt(args.Get("--vocab-size"), "--vocab-size"));
    }
    if (args.HasValue("--min-pair-frequency")) {
      options.min_pair_frequency = static_cast<std::size_t>(
          nsp::cli::ParseInt(args.Get("--min-pair-frequency"), "--min-pair-frequency"));
    }
    if (args.HasFlag("--no-lowercase")) {
      options.normalizer_options.lowercase = false;
    }
    if (args.HasFlag("--no-collapse-whitespace")) {
      options.normalizer_options.collapse_whitespace = false;
    }
    if (args.HasFlag("--no-dummy-prefix")) {
      options.normalizer_options.add_dummy_prefix = false;
    }

    nsp::BpeTrainer trainer(options);
    nsp::Model model = trainer.TrainFromFiles(args.positionals);
    model.Save(args.Get("--model"));

    std::cout << "trained model: " << args.Get("--model") << "\n";
    std::cout << "vocabulary size: " << model.vocabulary.Size() << "\n";
    std::cout << "merge count: " << model.merges.size() << "\n";
    std::cout << "special ids: "
              << "unk=" << model.vocabulary.unk_id() << ", "
              << "bos=" << model.vocabulary.bos_id() << ", "
              << "eos=" << model.vocabulary.eos_id() << ", "
              << "pad=" << model.vocabulary.pad_id() << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "nsp_train error: " << ex.what() << "\n";
    return 1;
  }
}
