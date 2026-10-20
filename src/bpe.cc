#include "accel/bpe.h"

#include "unicode_cats.inc"

#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace accel {
namespace {

void SkipWs(const std::string& s, size_t& i) {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) {
    ++i;
  }
}

[[noreturn]] void BadJson(const std::string& what, size_t i) {
  throw std::runtime_error(what + " at " + std::to_string(i));
}

void Expect(const std::string& s, size_t& i, char want) {
  SkipWs(s, i);
  if (i >= s.size() || s[i] != want) {
    BadJson(std::string("tokenizer json expected ") + want, i);
  }
  ++i;
}

void AppendUtf8(std::string& out, char32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

bool DecodeOne(const std::string& s, size_t i, char32_t& cp, size_t& n) {
  if (i >= s.size()) {
    return false;
  }
  const auto* b = reinterpret_cast<const unsigned char*>(s.data());
  const unsigned char c = b[i];
  if (c < 0x80) {
    cp = c;
    n = 1;
    return true;
  }
  size_t need = 0;
  char32_t value = 0;
  if ((c & 0xE0) == 0xC0) {
    need = 2;
    value = c & 0x1F;
  } else if ((c & 0xF0) == 0xE0) {
    need = 3;
    value = c & 0x0F;
  } else if ((c & 0xF8) == 0xF0) {
    need = 4;
    value = c & 0x07;
  } else {
    return false;
  }
  if (i + need > s.size()) {
    return false;
  }
  for (size_t k = 1; k < need; ++k) {
    if ((b[i + k] & 0xC0) != 0x80) {
      return false;
    }
    value = (value << 6) | (b[i + k] & 0x3F);
  }
  if (value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF) || (need == 2 && value < 0x80) ||
      (need == 3 && value < 0x800) || (need == 4 && value < 0x10000)) {
    return false;
  }
  cp = value;
  n = need;
  return true;
}

std::string LossyUtf8(const std::string& bytes) {
  std::string out;
  size_t i = 0;
  while (i < bytes.size()) {
    char32_t cp = 0;
    size_t n = 0;
    if (DecodeOne(bytes, i, cp, n)) {
      AppendUtf8(out, cp);
      i += n;
    } else {
      AppendUtf8(out, 0xFFFD);
      ++i;
    }
  }
  return out;
}

std::string ParseString(const std::string& s, size_t& i) {
  SkipWs(s, i);
  if (i >= s.size() || s[i] != '"') {
    BadJson("tokenizer json expected a string", i);
  }
  ++i;
  std::string out;
  while (i < s.size()) {
    const unsigned char c = static_cast<unsigned char>(s[i++]);
    if (c == '"') {
      return out;
    }
    if (c != '\\') {
      out.push_back(static_cast<char>(c));
      continue;
    }
    if (i >= s.size()) {
      BadJson("tokenizer json bad escape", i);
    }
    const char e = s[i++];
    switch (e) {
      case '"':
      case '\\':
      case '/':
        out.push_back(e);
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      case 'u': {
        if (i + 4 > s.size()) {
          BadJson("tokenizer json bad unicode escape", i);
        }
        int cp = 0;
        for (int k = 0; k < 4; ++k) {
          const char h = s[i++];
          cp <<= 4;
          if (h >= '0' && h <= '9') {
            cp += h - '0';
          } else if (h >= 'a' && h <= 'f') {
            cp += h - 'a' + 10;
          } else if (h >= 'A' && h <= 'F') {
            cp += h - 'A' + 10;
          } else {
            BadJson("tokenizer json bad unicode escape", i);
          }
        }
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
          i += 2;
          int lo = 0;
          for (int k = 0; k < 4; ++k) {
            const char h = s[i++];
            lo <<= 4;
            if (h >= '0' && h <= '9') {
              lo += h - '0';
            } else if (h >= 'a' && h <= 'f') {
              lo += h - 'a' + 10;
            } else if (h >= 'A' && h <= 'F') {
              lo += h - 'A' + 10;
            } else {
              BadJson("tokenizer json bad surrogate", i);
            }
          }
          cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
        }
        AppendUtf8(out, static_cast<char32_t>(cp));
        break;
      }
      default:
        BadJson("tokenizer json bad escape", i);
    }
  }
  BadJson("tokenizer json unterminated string", i);
}

void SkipValue(const std::string& s, size_t& i);

void SkipObjectOrArray(const std::string& s, size_t& i, char open, char close) {
  Expect(s, i, open);
  SkipWs(s, i);
  if (i < s.size() && s[i] == close) {
    ++i;
    return;
  }
  while (i < s.size()) {
    if (open == '{') {
      ParseString(s, i);
      Expect(s, i, ':');
    }
    SkipValue(s, i);
    SkipWs(s, i);
    if (i < s.size() && s[i] == ',') {
      ++i;
      continue;
    }
    if (i < s.size() && s[i] == close) {
      ++i;
      return;
    }
    BadJson("tokenizer json bad container", i);
  }
  BadJson("tokenizer json truncated container", i);
}

void SkipValue(const std::string& s, size_t& i) {
  SkipWs(s, i);
  if (i >= s.size()) {
    BadJson("tokenizer json truncated value", i);
  }
  const char c = s[i];
  if (c == '"') {
    ParseString(s, i);
    return;
  }
  if (c == '{') {
    SkipObjectOrArray(s, i, '{', '}');
    return;
  }
  if (c == '[') {
    SkipObjectOrArray(s, i, '[', ']');
    return;
  }
  if (s.compare(i, 4, "true") == 0 || s.compare(i, 4, "null") == 0) {
    i += 4;
    return;
  }
  if (s.compare(i, 5, "false") == 0) {
    i += 5;
    return;
  }
  if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
    if (c == '-') {
      ++i;
    }
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
      ++i;
    }
    if (i < s.size() && s[i] == '.') {
      ++i;
      while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        ++i;
      }
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
      ++i;
      if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
        ++i;
      }
      while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        ++i;
      }
    }
    return;
  }
  BadJson("tokenizer json bad value", i);
}

bool TryLiteral(const std::string& s, size_t& i, const char* word) {
  SkipWs(s, i);
  const size_t n = std::strlen(word);
  if (s.compare(i, n, word) == 0) {
    i += n;
    return true;
  }
  return false;
}

bool ParseBool(const std::string& s, size_t& i) {
  if (TryLiteral(s, i, "true")) {
    return true;
  }
  if (TryLiteral(s, i, "false")) {
    return false;
  }
  BadJson("tokenizer json expected a bool", i);
}

int64_t ParseInt(const std::string& s, size_t& i) {
  SkipWs(s, i);
  int sign = 1;
  if (i < s.size() && s[i] == '-') {
    sign = -1;
    ++i;
  }
  if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) {
    BadJson("tokenizer json expected an int", i);
  }
  int64_t value = 0;
  while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
    value = value * 10 + (s[i] - '0');
    ++i;
  }
  return sign * value;
}

bool InRanges(char32_t cp, const uint32_t* pairs, int count) {
  int lo = 0;
  int hi = count;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    const uint32_t start = pairs[mid * 2];
    const uint32_t end = pairs[mid * 2 + 1];
    if (cp < start) {
      hi = mid;
    } else if (cp > end) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

bool IsLetter(char32_t cp) { return InRanges(cp, k_letter_ranges, k_letter_ranges_count); }
bool IsNumber(char32_t cp) { return InRanges(cp, k_number_ranges, k_number_ranges_count); }

bool IsUnicodeSpace(char32_t cp) {
  return cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D || cp == 0x20 || cp == 0x85 ||
         cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
         cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

struct View {
  std::string utf8;
  std::vector<char32_t> cps;
  std::vector<uint32_t> byte_at;
};

View DecodeUtf8(std::string_view bytes) {
  View view;
  view.utf8.assign(bytes.data(), bytes.size());
  view.byte_at.push_back(0);
  size_t i = 0;
  while (i < view.utf8.size()) {
    char32_t cp = 0;
    size_t n = 0;
    if (!DecodeOne(view.utf8, i, cp, n)) {
      throw std::runtime_error("prompt is not utf-8");
    }
    view.cps.push_back(cp);
    i += n;
    view.byte_at.push_back(static_cast<uint32_t>(i));
  }
  return view;
}

size_t MatchContraction(const std::vector<char32_t>& cps, size_t i, size_t limit) {
  static const char* kWords[] = {"'re", "'ve", "'ll", "'s", "'t", "'m", "'d"};
  for (const char* word : kWords) {
    const size_t n = std::strlen(word);
    if (i + n > limit) {
      continue;
    }
    bool ok = true;
    for (size_t k = 0; k < n; ++k) {
      if (cps[i + k] != static_cast<unsigned char>(word[k])) {
        ok = false;
        break;
      }
    }
    if (ok) {
      return i + n;
    }
  }
  return i;
}

size_t MatchRun(const std::vector<char32_t>& cps, size_t i, size_t limit, bool (*kind)(char32_t)) {
  if (i < limit && cps[i] == U' ' && i + 1 < limit && kind(cps[i + 1])) {
    size_t j = i + 1;
    while (j < limit && kind(cps[j])) {
      ++j;
    }
    return j;
  }
  if (i < limit && kind(cps[i])) {
    size_t j = i;
    while (j < limit && kind(cps[j])) {
      ++j;
    }
    return j;
  }
  return i;
}

bool IsOther(char32_t cp) { return !IsUnicodeSpace(cp) && !IsLetter(cp) && !IsNumber(cp); }

size_t MatchTrailingSpace(const std::vector<char32_t>& cps, size_t i, size_t limit) {
  if (i >= limit || !IsUnicodeSpace(cps[i])) {
    return i;
  }
  size_t j = i;
  while (j < limit && IsUnicodeSpace(cps[j])) {
    ++j;
  }
  if (j == limit) {
    return j;
  }
  if (j - i >= 2) {
    return j - 1;
  }
  return i;
}

size_t MatchSpace(const std::vector<char32_t>& cps, size_t i, size_t limit) {
  if (i >= limit || !IsUnicodeSpace(cps[i])) {
    return i;
  }
  while (i < limit && IsUnicodeSpace(cps[i])) {
    ++i;
  }
  return i;
}

size_t MatchRegex(const std::vector<char32_t>& cps, size_t i, size_t limit) {
  size_t end = MatchContraction(cps, i, limit);
  if (end > i) {
    return end;
  }
  end = MatchRun(cps, i, limit, IsLetter);
  if (end > i) {
    return end;
  }
  end = MatchRun(cps, i, limit, IsNumber);
  if (end > i) {
    return end;
  }
  end = MatchRun(cps, i, limit, IsOther);
  if (end > i) {
    return end;
  }
  end = MatchTrailingSpace(cps, i, limit);
  if (end > i) {
    return end;
  }
  return MatchSpace(cps, i, limit);
}

}  // namespace

Bpe::Bpe() {
  int listed[256];
  int count = 0;
  for (int b = '!'; b <= '~'; ++b) {
    listed[count++] = b;
  }
  for (int b = 0xA1; b <= 0xAC; ++b) {
    listed[count++] = b;
  }
  for (int b = 0xAE; b <= 0xFF; ++b) {
    listed[count++] = b;
  }
  char seen[256] = {};
  for (int i = 0; i < count; ++i) {
    seen[listed[i]] = 1;
  }
  int codes[256];
  int nlist = count;
  for (int i = 0; i < count; ++i) {
    codes[i] = listed[i];
  }
  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    if (!seen[b]) {
      listed[nlist] = b;
      codes[nlist] = 256 + extra;
      ++nlist;
      ++extra;
    }
  }
  for (int i = 0; i < 256; ++i) {
    byte_id_[i] = -1;
    const char32_t cp = static_cast<char32_t>(codes[i]);
    char_to_byte_[cp] = static_cast<uint8_t>(listed[i]);
    byte_token_[listed[i]].clear();
    AppendUtf8(byte_token_[listed[i]], cp);
  }
}

void Bpe::Load(const std::string& path) {
  loaded_ = false;
  vocab_.clear();
  id_to_token_.clear();
  known_.clear();
  special_.clear();
  added_.clear();
  merges_.clear();
  for (int i = 0; i < 256; ++i) {
    byte_id_[i] = -1;
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open tokenizer");
  }
  const std::string file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  size_t i = 0;
  Expect(file, i, '{');
  bool saw_normalizer = false;
  bool saw_post = false;
  bool saw_pre = false;
  bool saw_decoder = false;
  bool saw_model = false;
  std::vector<std::pair<std::string, std::string>> raw_merges;
  std::vector<int32_t> special_ids;

  auto take_null = [&](const char* name) {
    if (!TryLiteral(file, i, "null")) {
      throw std::runtime_error(std::string(name) + " is not supported");
    }
  };

  while (true) {
    SkipWs(file, i);
    if (i < file.size() && file[i] == '}') {
      ++i;
      break;
    }
    const std::string key = ParseString(file, i);
    Expect(file, i, ':');
    SkipWs(file, i);
    if (key == "added_tokens") {
      Expect(file, i, '[');
      SkipWs(file, i);
      if (i < file.size() && file[i] != ']') {
        while (true) {
          Expect(file, i, '{');
          std::string content;
          int64_t id = -1;
          bool special = false;
          bool saw_content = false;
          bool saw_id = false;
          while (true) {
            SkipWs(file, i);
            if (file[i] == '}') {
              ++i;
              break;
            }
            const std::string field = ParseString(file, i);
            Expect(file, i, ':');
            if (field == "content") {
              content = ParseString(file, i);
              saw_content = true;
            } else if (field == "id") {
              id = ParseInt(file, i);
              saw_id = true;
            } else if (field == "special") {
              special = ParseBool(file, i);
            } else {
              SkipValue(file, i);
            }
            SkipWs(file, i);
            if (file[i] == ',') {
              ++i;
              continue;
            }
            if (file[i] == '}') {
              ++i;
              break;
            }
            BadJson("tokenizer json bad added token", i);
          }
          if (!saw_content || !saw_id || id < 0 || id > 2000000) {
            throw std::runtime_error("tokenizer added token is invalid");
          }
          added_.push_back(Added{content, static_cast<int32_t>(id)});
          if (special) {
            special_ids.push_back(static_cast<int32_t>(id));
          }
          SkipWs(file, i);
          if (file[i] == ',') {
            ++i;
            continue;
          }
          if (file[i] == ']') {
            ++i;
            break;
          }
          BadJson("tokenizer json bad added token list", i);
        }
      } else {
        ++i;
      }
    } else if (key == "normalizer") {
      take_null("normalizer");
      saw_normalizer = true;
    } else if (key == "post_processor") {
      take_null("post processor");
      saw_post = true;
    } else if (key == "pre_tokenizer") {
      std::vector<std::string> order;
      Expect(file, i, '{');
      while (true) {
        SkipWs(file, i);
        if (file[i] == '}') {
          ++i;
          break;
        }
        const std::string field = ParseString(file, i);
        Expect(file, i, ':');
        if (field == "type") {
          if (ParseString(file, i) != "Sequence") {
            throw std::runtime_error("tokenizer pretokenizer is not a sequence");
          }
        } else if (field == "pretokenizers") {
          Expect(file, i, '[');
          while (true) {
            SkipWs(file, i);
            if (file[i] == ']') {
              ++i;
              break;
            }
            Expect(file, i, '{');
            std::string type;
            bool individual = false;
            bool saw_individual = false;
            bool prefix = false;
            bool saw_prefix = false;
            bool use_regex = true;
            while (true) {
              SkipWs(file, i);
              if (file[i] == '}') {
                ++i;
                break;
              }
              const std::string sub = ParseString(file, i);
              Expect(file, i, ':');
              if (sub == "type") {
                type = ParseString(file, i);
              } else if (sub == "individual_digits") {
                individual = ParseBool(file, i);
                saw_individual = true;
              } else if (sub == "add_prefix_space") {
                prefix = ParseBool(file, i);
                saw_prefix = true;
              } else if (sub == "use_regex") {
                use_regex = ParseBool(file, i);
              } else {
                SkipValue(file, i);
              }
              SkipWs(file, i);
              if (file[i] == ',') {
                ++i;
                continue;
              }
              if (file[i] == '}') {
                ++i;
                break;
              }
              BadJson("tokenizer json bad pretokenizer", i);
            }
            if (type == "Digits") {
              if (!saw_individual || !individual) {
                throw std::runtime_error("tokenizer digits split is not per digit");
              }
            } else if (type == "ByteLevel") {
              if ((saw_prefix && prefix) || !use_regex) {
                throw std::runtime_error("tokenizer byte-level settings are not supported");
              }
            } else {
              throw std::runtime_error("tokenizer pretokenizer " + type + " is not supported");
            }
            order.push_back(type);
            SkipWs(file, i);
            if (file[i] == ',') {
              ++i;
              continue;
            }
            if (file[i] == ']') {
              ++i;
              break;
            }
            BadJson("tokenizer json bad pretokenizer list", i);
          }
        } else {
          SkipValue(file, i);
        }
        SkipWs(file, i);
        if (file[i] == ',') {
          ++i;
          continue;
        }
        if (file[i] == '}') {
          ++i;
          break;
        }
        BadJson("tokenizer json bad pretokenizer", i);
      }
      if (order.size() != 2 || order[0] != "Digits" || order[1] != "ByteLevel") {
        throw std::runtime_error("tokenizer pretokenizer order is not supported");
      }
      saw_pre = true;
    } else if (key == "decoder") {
      Expect(file, i, '{');
      bool byte = false;
      while (true) {
        SkipWs(file, i);
        if (file[i] == '}') {
          ++i;
          break;
        }
        const std::string field = ParseString(file, i);
        Expect(file, i, ':');
        if (field == "type") {
          byte = ParseString(file, i) == "ByteLevel";
        } else {
          SkipValue(file, i);
        }
        SkipWs(file, i);
        if (file[i] == ',') {
          ++i;
          continue;
        }
        if (file[i] == '}') {
          ++i;
          break;
        }
        BadJson("tokenizer json bad decoder", i);
      }
      if (!byte) {
        throw std::runtime_error("tokenizer decoder is not byte-level");
      }
      saw_decoder = true;
    } else if (key == "model") {
      Expect(file, i, '{');
      while (true) {
        SkipWs(file, i);
        if (file[i] == '}') {
          ++i;
          break;
        }
        const std::string field = ParseString(file, i);
        Expect(file, i, ':');
        if (field == "type") {
          if (ParseString(file, i) != "BPE") {
            throw std::runtime_error("tokenizer model is not bpe");
          }
        } else if (field == "dropout" || field == "unk_token" || field == "continuing_subword_prefix" ||
                   field == "end_of_word_suffix") {
          take_null(field.c_str());
        } else if (field == "fuse_unk" || field == "byte_fallback" || field == "ignore_merges") {
          if (ParseBool(file, i)) {
            throw std::runtime_error("tokenizer " + field + " is not supported");
          }
        } else if (field == "vocab") {
          Expect(file, i, '{');
          SkipWs(file, i);
          if (i < file.size() && file[i] != '}') {
            while (true) {
              const std::string token = ParseString(file, i);
              Expect(file, i, ':');
              const int64_t id = ParseInt(file, i);
              if (id < 0 || id > 2000000) {
                throw std::runtime_error("tokenizer vocab id is out of range");
              }
              const auto placed = vocab_.emplace(token, static_cast<int32_t>(id));
              if (!placed.second) {
                throw std::runtime_error("tokenizer vocab has a duplicate token");
              }
              if (static_cast<size_t>(id) >= known_.size()) {
                known_.resize(static_cast<size_t>(id) + 1);
                id_to_token_.resize(static_cast<size_t>(id) + 1);
              }
              if (known_[static_cast<size_t>(id)]) {
                throw std::runtime_error("tokenizer vocab has a duplicate id");
              }
              known_[static_cast<size_t>(id)] = 1;
              id_to_token_[static_cast<size_t>(id)] = token;
              SkipWs(file, i);
              if (file[i] == ',') {
                ++i;
                continue;
              }
              if (file[i] == '}') {
                ++i;
                break;
              }
              BadJson("tokenizer json bad vocab", i);
            }
          } else {
            ++i;
          }
        } else if (field == "merges") {
          Expect(file, i, '[');
          SkipWs(file, i);
          if (i < file.size() && file[i] != ']') {
            while (true) {
              const std::string line = ParseString(file, i);
              const size_t space = line.find(' ');
              if (space == std::string::npos || line.find(' ', space + 1) != std::string::npos) {
                throw std::runtime_error("tokenizer merge is not a pair");
              }
              raw_merges.emplace_back(line.substr(0, space), line.substr(space + 1));
              SkipWs(file, i);
              if (file[i] == ',') {
                ++i;
                continue;
              }
              if (file[i] == ']') {
                ++i;
                break;
              }
              BadJson("tokenizer json bad merges", i);
            }
          } else {
            ++i;
          }
        } else {
          SkipValue(file, i);
        }
        SkipWs(file, i);
        if (file[i] == ',') {
          ++i;
          continue;
        }
        if (file[i] == '}') {
          ++i;
          break;
        }
        BadJson("tokenizer json bad model", i);
      }
      saw_model = true;
    } else {
      SkipValue(file, i);
    }
    SkipWs(file, i);
    if (i < file.size() && file[i] == ',') {
      ++i;
      continue;
    }
    if (i < file.size() && file[i] == '}') {
      ++i;
      break;
    }
    BadJson("tokenizer json bad object", i);
  }

  if (!saw_normalizer || !saw_post || !saw_pre || !saw_decoder || !saw_model || vocab_.empty() || raw_merges.empty()) {
    throw std::runtime_error("tokenizer json is missing bpe data");
  }
  for (size_t n = 0; n < known_.size(); ++n) {
    if (!known_[n]) {
      throw std::runtime_error("tokenizer vocab has a hole");
    }
  }
  special_.assign(id_to_token_.size(), 0);
  for (int32_t id : special_ids) {
    if (id < 0 || static_cast<size_t>(id) >= special_.size()) {
      throw std::runtime_error("tokenizer special id is outside the vocab");
    }
    special_[static_cast<size_t>(id)] = 1;
  }
  for (const Added& added : added_) {
    const auto it = vocab_.find(added.text);
    if (it == vocab_.end() || it->second != added.id) {
      throw std::runtime_error("tokenizer added token is not in the vocab");
    }
  }
  for (int b = 0; b < 256; ++b) {
    const auto it = vocab_.find(byte_token_[b]);
    if (it != vocab_.end()) {
      byte_id_[b] = it->second;
    }
  }
  uint32_t rank = 0;
  for (const auto& pair : raw_merges) {
    const auto left = vocab_.find(pair.first);
    const auto right = vocab_.find(pair.second);
    if (left == vocab_.end() || right == vocab_.end()) {
      throw std::runtime_error("tokenizer merge token is outside the vocab");
    }
    const auto joined = vocab_.find(pair.first + pair.second);
    if (joined == vocab_.end()) {
      throw std::runtime_error("tokenizer merge result is outside the vocab");
    }
    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(left->second)) << 32) |
                         static_cast<uint32_t>(right->second);
    const Merge merge{rank, static_cast<uint32_t>(joined->second)};
    const auto placed = merges_.emplace(key, merge);
    if (!placed.second && rank < placed.first->second.rank) {
      placed.first->second = merge;
    }
    ++rank;
  }
  loaded_ = true;
}

size_t Bpe::MatchAdded(const std::string& text, size_t pos, int32_t& id) const {
  size_t best = 0;
  int32_t best_id = -1;
  for (const Added& added : added_) {
    if (added.text.size() <= best || pos + added.text.size() > text.size()) {
      continue;
    }
    if (std::memcmp(text.data() + pos, added.text.data(), added.text.size()) == 0) {
      best = added.text.size();
      best_id = added.id;
    }
  }
  if (best == 0) {
    return 0;
  }
  id = best_id;
  return best;
}

void Bpe::MergeWord(const std::vector<int32_t>& chars, std::vector<int32_t>& out) const {
  std::vector<int32_t> sym = chars;
  while (sym.size() >= 2) {
    int best_rank = std::numeric_limits<int>::max();
    int best_i = -1;
    uint32_t best_id = 0;
    for (int n = 0; n + 1 < static_cast<int>(sym.size()); ++n) {
      const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(sym[static_cast<size_t>(n)])) << 32) |
                           static_cast<uint32_t>(sym[static_cast<size_t>(n + 1)]);
      const auto it = merges_.find(key);
      if (it != merges_.end() && static_cast<int>(it->second.rank) < best_rank) {
        best_rank = static_cast<int>(it->second.rank);
        best_i = n;
        best_id = it->second.id;
      }
    }
    if (best_i < 0) {
      break;
    }
    sym[static_cast<size_t>(best_i)] = static_cast<int32_t>(best_id);
    sym.erase(sym.begin() + best_i + 1);
  }
  out.insert(out.end(), sym.begin(), sym.end());
}

std::vector<int32_t> Bpe::EncodePiece(std::string_view piece) const {
  std::vector<int32_t> out;
  if (piece.empty()) {
    return out;
  }
  const View view = DecodeUtf8(piece);
  size_t cursor = 0;
  while (cursor < view.cps.size()) {
    size_t run = cursor;
    if (IsNumber(view.cps[cursor])) {
      run = cursor + 1;
    } else {
      while (run < view.cps.size() && !IsNumber(view.cps[run])) {
        ++run;
      }
    }
    size_t at = cursor;
    while (at < run) {
      const size_t end = MatchRegex(view.cps, at, run);
      if (end <= at) {
        throw std::runtime_error("tokenizer regex stalled");
      }
      std::vector<int32_t> chars;
      chars.reserve(view.byte_at[end] - view.byte_at[at]);
      for (uint32_t byte_i = view.byte_at[at]; byte_i < view.byte_at[end]; ++byte_i) {
        const auto byte = static_cast<unsigned char>(view.utf8[byte_i]);
        const int32_t id = byte_id_[byte];
        // This vocab omits some GPT-2 byte atoms. tokenizers drops those
        // symbols and still merges the rest of the same piece.
        if (id >= 0) {
          chars.push_back(id);
        }
      }
      if (!chars.empty()) {
        MergeWord(chars, out);
      }
      at = end;
    }
    cursor = run;
  }
  return out;
}

std::vector<int32_t> Bpe::Encode(const std::string& text) const {
  if (!loaded_) {
    throw std::runtime_error("tokenizer is not loaded");
  }
  std::vector<int32_t> out;
  size_t pos = 0;
  while (pos < text.size()) {
    int32_t added_id = -1;
    const size_t taken = MatchAdded(text, pos, added_id);
    if (taken > 0) {
      out.push_back(added_id);
      pos += taken;
      continue;
    }
    size_t gap = pos + 1;
    while (gap < text.size()) {
      int32_t ignore = -1;
      if (MatchAdded(text, gap, ignore) > 0) {
        break;
      }
      ++gap;
    }
    const std::vector<int32_t> piece = EncodePiece(std::string_view(text.data() + pos, gap - pos));
    out.insert(out.end(), piece.begin(), piece.end());
    pos = gap;
  }
  return out;
}

std::string Bpe::Decode(const int32_t* ids, int32_t count, bool skip_special) const {
  if (!loaded_) {
    throw std::runtime_error("tokenizer is not loaded");
  }
  if (count < 0 || (count > 0 && ids == nullptr)) {
    throw std::runtime_error("invalid decode");
  }
  std::string bytes;
  for (int32_t n = 0; n < count; ++n) {
    const int32_t id = ids[n];
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size() || !known_[static_cast<size_t>(id)]) {
      throw std::runtime_error("token id is outside the vocabulary");
    }
    if (skip_special && special_[static_cast<size_t>(id)]) {
      continue;
    }
    const std::string& token = id_to_token_[static_cast<size_t>(id)];
    std::string mapped;
    bool ok = true;
    size_t i = 0;
    while (i < token.size()) {
      char32_t cp = 0;
      size_t len = 0;
      if (!DecodeOne(token, i, cp, len)) {
        ok = false;
        break;
      }
      const auto it = char_to_byte_.find(cp);
      if (it == char_to_byte_.end()) {
        ok = false;
        break;
      }
      mapped.push_back(static_cast<char>(it->second));
      i += len;
    }
    bytes += ok ? mapped : token;
  }
  return LossyUtf8(bytes);
}

}  // namespace accel
