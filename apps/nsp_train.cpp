#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "nanosentencepiece/bpe_trainer.hpp"
#include "nanosentencepiece/cli.hpp"
#include "nanosentencepiece/model.hpp"
#include "nanosentencepiece/unigram_trainer.hpp"
#include "nanosentencepiece/version.hpp"

namespace nsp = nanosentencepiece;

namespace {

void PrintUsage() {
  std::cout
      << "nsp_train " << nsp::kVersion << "\n"
      << "Usage:\n"
      << "  nsp_train --model model.nsp [--model-type bpe|unigram] --vocab-size 128\n"
      << "            [--min-pair-frequency 2] [--min-piece-frequency 2]\n"
      << "            [--max-piece-length 8] [--unigram-iterations 4]\n"
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

    const std::string model_type = args.Get("--model-type", "bpe");
    nsp::Model model;
    if (model_type == "bpe") {
      nsp::BpeTrainer trainer(options);
      model = trainer.TrainFromFiles(args.positionals);
    } else if (model_type == "unigram") {
      nsp::UnigramTrainerOptions unigram_options;
      unigram_options.vocab_size = options.vocab_size;
      unigram_options.normalizer_options = options.normalizer_options;
      unigram_options.special_tokens = options.special_tokens;

      if (args.HasValue("--min-piece-frequency")) {
        unigram_options.min_piece_frequency = static_cast<std::size_t>(
            nsp::cli::ParseInt(args.Get("--min-piece-frequency"), "--min-piece-frequency"));
      }
      if (args.HasValue("--max-piece-length")) {
        unigram_options.max_piece_length = static_cast<std::size_t>(
            nsp::cli::ParseInt(args.Get("--max-piece-length"), "--max-piece-length"));
      }
      if (args.HasValue("--unigram-iterations")) {
        unigram_options.num_iterations = static_cast<std::size_t>(
            nsp::cli::ParseInt(args.Get("--unigram-iterations"), "--unigram-iterations"));
      }

      nsp::UnigramTrainer trainer(unigram_options);
      model = trainer.TrainFromFiles(args.positionals);
    } else {
      throw std::runtime_error("unknown --model-type: " + model_type);
    }
    model.Save(args.Get("--model"));

    std::cout << "trained model: " << args.Get("--model") << "\n";
    std::cout << "model type: " << nsp::ToString(model.metadata.model_type) << "\n";
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
