#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace accel {

// Hugging Face BPE used by SmolLM2: digit splits, then the GPT-2 byte-level
// regex, then merges. Special tokens are taken out of the text first.
class Bpe {
 public:
  Bpe();
  void Load(const std::string& path);
  bool loaded() const { return loaded_; }

  std::vector<int32_t> Encode(const std::string& text) const;
  std::string Decode(const int32_t* ids, int32_t count, bool skip_special) const;

 private:
  struct Added {
    std::string text;
    int32_t id = 0;
  };
  struct Merge {
    uint32_t rank = 0;
    uint32_t id = 0;
  };

  std::vector<int32_t> EncodePiece(std::string_view piece) const;
  void MergeWord(const std::vector<int32_t>& chars, std::vector<int32_t>& out) const;
  size_t MatchAdded(const std::string& text, size_t pos, int32_t& id) const;

  bool loaded_ = false;
  std::vector<std::string> id_to_token_;
  std::vector<char> known_;
  std::vector<char> special_;
  std::vector<Added> added_;
  std::unordered_map<std::string, int32_t> vocab_;
  std::unordered_map<uint64_t, Merge> merges_;
  std::string byte_token_[256];
  int32_t byte_id_[256]{};
  std::unordered_map<char32_t, uint8_t> char_to_byte_;
};

}  // namespace accel
