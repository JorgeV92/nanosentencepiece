#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "nanosentencepiece/bpe_trainer.hpp"
#include "nanosentencepiece/model.hpp"
#include "nanosentencepiece/normalization.hpp"
#include "nanosentencepiece/processor.hpp"
#include "nanosentencepiece/tokenizer.hpp"
#include "nanosentencepiece/unigram_trainer.hpp"

namespace nsp = nanosentencepiece;

namespace {

std::string TempModelPath(const std::string& filename) {
  return (std::filesystem::temp_directory_path() / filename).string();
}

TEST(NormalizationTest, AppliesAsciiNormalizationAndWhitespaceEscaping) {
  nsp::NormalizerOptions options;
  options.lowercase = true;
  options.collapse_whitespace = true;
  nsp::Normalizer normalizer(options);

  const std::string normalized = normalizer.Normalize("  HeLLo\t\tWORLD  ");
  EXPECT_EQ(normalized, "hello world");

  const std::string escaped = normalizer.EscapeWhitespace(normalized);
  EXPECT_EQ(escaped, "▁hello▁world");

  const std::string restored = normalizer.RestoreWhitespace(escaped);
  EXPECT_EQ(restored, "hello world");
}

TEST(BpeTrainerTest, LearnsMostFrequentMerge) {
  nsp::BpeTrainerOptions options;
  options.vocab_size = 16;
  options.min_pair_frequency = 2;
  options.normalizer_options.lowercase = false;
  options.normalizer_options.collapse_whitespace = false;
  options.normalizer_options.add_dummy_prefix = false;

  nsp::BpeTrainer trainer(options);
  const nsp::Model model = trainer.TrainFromLines({"abab", "abab"});
  ASSERT_FALSE(model.merges.empty());
  EXPECT_EQ(model.merges.front().merged, "ab");
}

TEST(BpeTrainerTest, PreservesLexicographicTieBreaks) {
  nsp::BpeTrainerOptions options;
  options.vocab_size = 16;
  options.min_pair_frequency = 1;
  options.normalizer_options.lowercase = false;
  options.normalizer_options.collapse_whitespace = false;
  options.normalizer_options.add_dummy_prefix = false;

  nsp::BpeTrainer trainer(options);
  const nsp::Model model = trainer.TrainFromLines({"ab", "ac"});

  ASSERT_FALSE(model.merges.empty());
  EXPECT_EQ(model.merges.front().left, "a");
  EXPECT_EQ(model.merges.front().right, "b");
  EXPECT_EQ(model.merges.front().merged, "ab");
}

TEST(TokenizerTest, RoundTripsNormalizedText) {
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
  EXPECT_EQ(decoded, "hello world");
}

TEST(TokenizerTest, FallsBackToUnknownPiece) {
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
  EXPECT_TRUE(saw_unk);
}

TEST(ProcessorTest, MatchesTokenizerBehaviorAndRetainsSharedModel) {
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
  EXPECT_EQ(processor_pieces, tokenizer_pieces);

  const auto processor_ids =
      processor.EncodeToIds("tokenizer compatibility check", nsp::EncodeOptions{true, true});
  const auto tokenizer_ids = tokenizer.EncodeToIds("tokenizer compatibility check", true, true);
  EXPECT_EQ(processor_ids, tokenizer_ids);

  EXPECT_EQ(processor.model_ptr().get(), shared_model.get());
}

TEST(ModelTest, SerializationPreservesSpecialIds) {
  nsp::BpeTrainerOptions options;
  options.vocab_size = 48;
  options.min_pair_frequency = 1;

  nsp::BpeTrainer trainer(options);
  nsp::Model model = trainer.TrainFromLines({"serialize me", "and load me back"});
  const std::string path = TempModelPath("nsp_test_model.nsp");
  model.Save(path);

  const nsp::Model loaded = nsp::Model::Load(path);
  nsp::Tokenizer tokenizer(loaded);
  const auto ids = tokenizer.EncodeToIds("serialize me", true, true);
  ASSERT_FALSE(ids.empty());
  EXPECT_EQ(ids.front(), loaded.vocabulary.bos_id());
  EXPECT_EQ(ids.back(), loaded.vocabulary.eos_id());
  EXPECT_TRUE(std::filesystem::remove(path));
}

TEST(UnigramTrainerTest, TrainsUnigramModelAndRoundTripsText) {
  nsp::UnigramTrainerOptions options;
  options.vocab_size = 32;
  options.max_piece_length = 8;
  options.min_piece_frequency = 1;
  options.num_iterations = 4;

  nsp::UnigramTrainer trainer(options);
  const nsp::Model model = trainer.TrainFromLines({
      "banana bandana banana",
      "banana band",
      "bandana banana",
  });

  EXPECT_EQ(model.metadata.model_type, nsp::ModelType::kUnigram);
  EXPECT_TRUE(model.IsUnigram());
  EXPECT_TRUE(model.merges.empty());

  nsp::SentencePieceProcessor processor(model);
  const auto pieces = processor.EncodeToPieces("banana bandana", nsp::EncodeOptions{true, true});
  EXPECT_FALSE(pieces.empty());
  EXPECT_EQ(processor.DecodePieces(pieces), "banana bandana");

  std::size_t unk_count = 0;
  for (const auto& piece : pieces) {
    if (piece == model.special_tokens.unk) {
      ++unk_count;
    }
  }
  EXPECT_EQ(unk_count, 0U);
}

TEST(UnigramTrainerTest, SerializationPreservesUnigramInference) {
  nsp::UnigramTrainerOptions options;
  options.vocab_size = 24;
  options.max_piece_length = 6;
  options.min_piece_frequency = 1;
  options.num_iterations = 3;

  nsp::UnigramTrainer trainer(options);
  nsp::Model model = trainer.TrainFromLines({
      "lowest low",
      "lower lowest",
      "low low lower",
  });

  const nsp::SentencePieceProcessor before_processor(model);
  const auto before = before_processor.EncodeToPieces("lowest low");

  const std::string path = TempModelPath("nsp_test_unigram_model.nsp");
  model.Save(path);

  const nsp::Model loaded = nsp::Model::Load(path);
  const nsp::SentencePieceProcessor after_processor(loaded);
  EXPECT_EQ(loaded.metadata.model_type, nsp::ModelType::kUnigram);
  EXPECT_EQ(after_processor.EncodeToPieces("lowest low"), before);
  EXPECT_EQ(after_processor.DecodePieces(before), "lowest low");
  EXPECT_TRUE(std::filesystem::remove(path));
}

TEST(ProcessorTest, RepeatedUnigramEncodesStayStableAcrossCopies) {
  nsp::UnigramTrainerOptions options;
  options.vocab_size = 32;
  options.max_piece_length = 8;
  options.min_piece_frequency = 1;
  options.num_iterations = 4;

  nsp::UnigramTrainer trainer(options);
  const nsp::Model model = trainer.TrainFromLines({
      "banana bandana banana",
      "banana band",
      "bandana banana",
  });

  const nsp::SentencePieceProcessor processor(model);
  const auto first = processor.EncodeToPieces("banana bandana");
  const auto second = processor.EncodeToPieces("banana bandana");
  EXPECT_EQ(second, first);

  const nsp::SentencePieceProcessor copied = processor;
  EXPECT_EQ(copied.EncodeToPieces("banana bandana"), first);
}

TEST(ProcessorTest, LoadsModelFromDisk) {
  nsp::BpeTrainerOptions options;
  options.vocab_size = 48;
  options.min_pair_frequency = 1;

  nsp::BpeTrainer trainer(options);
  nsp::Model model = trainer.TrainFromLines({"load through processor", "decode through processor"});
  const std::string path = TempModelPath("nsp_test_processor_model.nsp");
  model.Save(path);

  const nsp::SentencePieceProcessor processor = nsp::SentencePieceProcessor::Load(path);
  const auto pieces = processor.EncodeToPieces("load through processor", nsp::EncodeOptions{true, true});
  EXPECT_FALSE(pieces.empty());
  EXPECT_EQ(processor.DecodePieces(pieces), "load through processor");
  EXPECT_TRUE(std::filesystem::remove(path));
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
