#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nanosentencepiece/bpe_trainer.hpp"
#include "nanosentencepiece/model.hpp"
#include "nanosentencepiece/normalization.hpp"
#include "nanosentencepiece/processor.hpp"
#include "nanosentencepiece/tokenizer.hpp"

namespace nsp = nanosentencepiece;

namespace {

struct TestFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw TestFailure(message);
  }
}

void ExpectEq(const std::string& actual, const std::string& expected, const std::string& message) {
  if (actual != expected) {
    throw TestFailure(message + " | expected=\"" + expected + "\" actual=\"" + actual + "\"");
  }
}

void ExpectEq(int actual, int expected, const std::string& message) {
  if (actual != expected) {
    throw TestFailure(message + " | expected=" + std::to_string(expected) +
                      " actual=" + std::to_string(actual));
  }
}

void TestNormalization() {
  nsp::NormalizerOptions options;
  options.lowercase = true;
  options.collapse_whitespace = true;
  nsp::Normalizer normalizer(options);

  const std::string normalized = normalizer.Normalize("  HeLLo\t\tWORLD  ");
  ExpectEq(normalized, "hello world", "ascii lowercase + whitespace collapse");

  const std::string escaped = normalizer.EscapeWhitespace(normalized);
  ExpectEq(escaped, "▁hello▁world", "whitespace should be reversible");

  const std::string restored = normalizer.RestoreWhitespace(escaped);
  ExpectEq(restored, "hello world", "restored text should match normalized text");
}

void TestBpeLearnsAbMerge() {
  nsp::BpeTrainerOptions options;
  options.vocab_size = 16;
  options.min_pair_frequency = 2;
  options.normalizer_options.lowercase = false;
  options.normalizer_options.collapse_whitespace = false;
  options.normalizer_options.add_dummy_prefix = false;

  nsp::BpeTrainer trainer(options);
  const nsp::Model model = trainer.TrainFromLines({"abab", "abab"});
  Expect(!model.merges.empty(), "trainer should learn at least one merge");
  ExpectEq(model.merges.front().merged, "ab", "first merge should be ab");
}

void TestRoundTrip() {
  nsp::BpeTrainerOptions options;
  options.vocab_size = 64;
  options.min_pair_frequency = 1;

  nsp::BpeTrainer trainer(options);
  const nsp::Model model = trainer.TrainFromLines({
      "Hello world",
      "hello tokenizer world",
      "small reversible tokenizer",
  });

  nsp::Tokenizer tokenizer(model);
  const auto pieces = tokenizer.EncodeToPieces("Hello   world", true, true);
  const std::string decoded = tokenizer.DecodePieces(pieces);
  ExpectEq(decoded, "hello world", "decode should round-trip normalized text");
}

void TestUnknownFallback() {
  nsp::BpeTrainerOptions options;
  options.vocab_size = 32;
  options.min_pair_frequency = 1;
  options.normalizer_options.lowercase = false;
  options.normalizer_options.collapse_whitespace = false;

  nsp::BpeTrainer trainer(options);
  const nsp::Model model = trainer.TrainFromLines({"abc abc"});
  nsp::Tokenizer tokenizer(model);

  const auto pieces = tokenizer.EncodeToPieces("abc zzz");
  bool saw_unk = false;
  for (const auto& piece : pieces) {
    if (piece == model.special_tokens.unk) {
      saw_unk = true;
      break;
    }
  }
  Expect(saw_unk, "unknown characters should map to <unk>");
}

void TestProcessorMatchesTokenizer() {
  nsp::BpeTrainerOptions options;
  options.vocab_size = 64;
  options.min_pair_frequency = 1;

  nsp::BpeTrainer trainer(options);
  const nsp::Model model = trainer.TrainFromLines({
      "processor api parity",
      "tokenizer compatibility check",
      "shared model path",
  });

  const auto shared_model = std::make_shared<const nsp::Model>(model);
  const nsp::SentencePieceProcessor processor(shared_model);
  const nsp::Tokenizer tokenizer(model);

  const auto processor_pieces =
      processor.EncodeToPieces("processor api parity", nsp::EncodeOptions{true, true});
  const auto tokenizer_pieces = tokenizer.EncodeToPieces("processor api parity", true, true);
  Expect(processor_pieces == tokenizer_pieces, "processor pieces should match tokenizer pieces");

  const auto processor_ids =
      processor.EncodeToIds("tokenizer compatibility check", nsp::EncodeOptions{true, true});
  const auto tokenizer_ids = tokenizer.EncodeToIds("tokenizer compatibility check", true, true);
  Expect(processor_ids == tokenizer_ids, "processor ids should match tokenizer ids");

  Expect(processor.model_ptr().get() == shared_model.get(), "processor should retain shared model");
}

void TestSerialization() {
  nsp::BpeTrainerOptions options;
  options.vocab_size = 48;
  options.min_pair_frequency = 1;

  nsp::BpeTrainer trainer(options);
  nsp::Model model = trainer.TrainFromLines({"serialize me", "and load me back"});
  const std::string path = "nsp_test_model.nsp";
  model.Save(path);

  const nsp::Model loaded = nsp::Model::Load(path);
  nsp::Tokenizer tokenizer(loaded);
  const auto ids = tokenizer.EncodeToIds("serialize me", true, true);
  ExpectEq(ids.front(), loaded.vocabulary.bos_id(), "BOS id should be stable after load");
  ExpectEq(ids.back(), loaded.vocabulary.eos_id(), "EOS id should be stable after load");
}

void TestProcessorLoad() {
  nsp::BpeTrainerOptions options;
  options.vocab_size = 48;
  options.min_pair_frequency = 1;

  nsp::BpeTrainer trainer(options);
  nsp::Model model = trainer.TrainFromLines({"load through processor", "decode through processor"});
  const std::string path = "nsp_test_processor_model.nsp";
  model.Save(path);

  const nsp::SentencePieceProcessor processor = nsp::SentencePieceProcessor::Load(path);
  const auto pieces = processor.EncodeToPieces("load through processor", nsp::EncodeOptions{true, true});
  Expect(!pieces.empty(), "processor load should produce pieces");
  ExpectEq(processor.DecodePieces(pieces), "load through processor",
           "processor decode should round-trip normalized text");
}

}  // namespace

int main() {
  const std::vector<std::pair<std::string, std::function<void()>>> tests = {
      {"TestNormalization", TestNormalization},
      {"TestBpeLearnsAbMerge", TestBpeLearnsAbMerge},
      {"TestRoundTrip", TestRoundTrip},
      {"TestUnknownFallback", TestUnknownFallback},
      {"TestProcessorMatchesTokenizer", TestProcessorMatchesTokenizer},
      {"TestSerialization", TestSerialization},
      {"TestProcessorLoad", TestProcessorLoad},
  };

  int failures = 0;
  for (const auto& [name, test] : tests) {
    try {
      test();
      std::cout << "[PASS] " << name << "\n";
    } catch (const std::exception& ex) {
      ++failures;
      std::cerr << "[FAIL] " << name << ": " << ex.what() << "\n";
    }
  }

  if (failures > 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }

  std::cout << "all tests passed\n";
  return 0;
}
