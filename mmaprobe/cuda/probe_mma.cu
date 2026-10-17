// probe_mma.cu - run ONE tensor-core mma.sync per query vector and return the raw FP32
// accumulator bits of D[0][0].
//
//   nvcc -O2 -arch=sm_89 -o probe_mma probe_mma.cu      # L4 / RTX 40xx  (needs nvcc >= 12.4 for e4m3 mma)
//   nvcc -O2 -arch=sm_75 -o probe_mma probe_mma.cu      # T4             (f16_k8 only; FP8 path is compiled out)
//
//   ./probe_mma --info
//   ./probe_mma <config> --selftest
//   ./probe_mma <config> in.bin out.bin
//
// configs: e4m3_k32  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32   (sm_89+)
//          f16_k8    mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32      (sm_75+)
//
// Wire format (little endian). in.bin : u32 n, then n records [a: K elems][b: K elems][c: u32],
// elements are 1 byte (e4m3) or 2 bytes (f16). out.bin: n x u32 (D[0][0] bits).
//
// A is a 16xK matrix whose row 0 is `a` and whose other rows are zero. B is Kx8 whose column 0 is
// `b`, others zero. C[0][0] = c, others zero. Fragment layouts follow the PTX ISA documentation
// for these shapes. They were written without access to a GPU: run --selftest first. It checks
// that A and B agree on the k index of every element, that C[0][0] lands in lane 0 / d0, and that
// nothing leaks between k positions.
//
// There is no compiler barrier problem here: the mma is inline PTX with volatile, one per warp,
// and the result is read straight from the D registers without any further arithmetic.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CK(x)                                                                                   \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(2);                                                                             \
    }                                                                                           \
  } while (0)

enum Cfg { E4M3_K32 = 0, F16_K8 = 1 };

struct CfgInfo { int K; int esize; };
static CfgInfo info_of(int cfg) { return cfg == E4M3_K32 ? CfgInfo{32, 1} : CfgInfo{8, 2}; }

__device__ __forceinline__ uint32_t elem(const uint8_t* base, int idx, int esize) {
  if (esize == 1) return base[idx];
  return (uint32_t)base[2 * idx] | ((uint32_t)base[2 * idx + 1] << 8);
}

__global__ void probe_kernel(int cfg, const uint8_t* __restrict__ in, uint32_t* __restrict__ out,
                             uint32_t n, uint32_t rec_bytes) {
  const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= n) return;
  const int lane = threadIdx.x & 31;
  const int gid = lane >> 2;
  const int tig = lane & 3;
  const uint8_t* rec = in + (size_t)warp * rec_bytes;

  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

  if (cfg == E4M3_K32) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && !defined(NO_FP8)
    const int K = 32, es = 1;
    const uint8_t* a = rec;
    const uint8_t* b = rec + K * es;
    uint32_t cbits;
    memcpy(&cbits, rec + 2 * K * es, 4);
    uint32_t ra[4], rb[2];
    for (int r = 0; r < 4; ++r) {
      uint32_t reg = 0;
      for (int j = 0; j < 4; ++j) {
        const int row = gid + ((r & 1) ? 8 : 0);
        const int col = tig * 4 + j + (r >= 2 ? 16 : 0);
        const uint32_t v = (row == 0) ? elem(a, col, es) : 0u;
        reg |= v << (8 * j);
      }
      ra[r] = reg;
    }
    for (int r = 0; r < 2; ++r) {
      uint32_t reg = 0;
      for (int j = 0; j < 4; ++j) {
        const int k = tig * 4 + j + (r == 1 ? 16 : 0);
        const int col = gid;
        const uint32_t v = (col == 0) ? elem(b, k, es) : 0u;
        reg |= v << (8 * j);
      }
      rb[r] = reg;
    }
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    if (gid == 0 && tig == 0) c0 = __uint_as_float(cbits);  // C[0][0]
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
    d0 = __uint_as_float(0xFFFFFFFFu);  // unsupported on this build; host reports it
#endif
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const int K = 8, es = 2;
    const uint8_t* a = rec;
    const uint8_t* b = rec + K * es;
    uint32_t cbits;
    memcpy(&cbits, rec + 2 * K * es, 4);
    uint32_t ra[2], rb[1];
    for (int r = 0; r < 2; ++r) {
      const int row = gid + (r == 1 ? 8 : 0);
      const uint32_t lo = (row == 0) ? elem(a, tig * 2 + 0, es) : 0u;
      const uint32_t hi = (row == 0) ? elem(a, tig * 2 + 1, es) : 0u;
      ra[r] = lo | (hi << 16);
    }
    {
      const uint32_t lo = (gid == 0) ? elem(b, tig * 2 + 0, es) : 0u;
      const uint32_t hi = (gid == 0) ? elem(b, tig * 2 + 1, es) : 0u;
      rb[0] = lo | (hi << 16);
    }
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    if (gid == 0 && tig == 0) c0 = __uint_as_float(cbits);
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(ra[0]), "r"(ra[1]), "r"(rb[0]), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
    d0 = __uint_as_float(0xFFFFFFFFu);
#endif
  }
  if (lane == 0) out[warp] = __float_as_uint(d0);  // lane 0, d0 holds D[0][0]
}

static std::vector<uint32_t> run_batch(int cfg, const std::vector<uint8_t>& records, uint32_t n) {
  const CfgInfo ci = info_of(cfg);
  const uint32_t rec_bytes = 2 * ci.K * ci.esize + 4;
  uint8_t* d_in = nullptr;
  uint32_t* d_out = nullptr;
  CK(cudaMalloc(&d_in, records.size()));
  CK(cudaMalloc(&d_out, sizeof(uint32_t) * (n ? n : 1)));
  CK(cudaMemcpy(d_in, records.data(), records.size(), cudaMemcpyHostToDevice));
  const uint64_t threads = (uint64_t)n * 32;
  const int block = 128;
  const uint64_t grid = (threads + block - 1) / block;
  if (n) probe_kernel<<<(unsigned)grid, block>>>(cfg, d_in, d_out, n, rec_bytes);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<uint32_t> out(n);
  if (n) CK(cudaMemcpy(out.data(), d_out, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost));
  cudaFree(d_in);
  cudaFree(d_out);
  return out;
}

static int selftest(int cfg) {
  const CfgInfo ci = info_of(cfg);
  const int K = ci.K;
  const uint32_t one_elem = cfg == E4M3_K32 ? 0x38u : 0x3C00u;  // 1.0
  const uint32_t one_f32 = 0x3F800000u;
  const uint32_t rec_bytes = 2 * K * ci.esize + 4;
  std::vector<std::vector<uint32_t>> as, bs;
  std::vector<uint32_t> cs, expect;
  auto zeros = [&] { return std::vector<uint32_t>(K, 0u); };
  // 1. c only
  as.push_back(zeros()); bs.push_back(zeros()); cs.push_back(one_f32); expect.push_back(one_f32);
  // 2. single matching k
  for (int k = 0; k < K; ++k) {
    auto a = zeros(), b = zeros(); a[k] = one_elem; b[k] = one_elem;
    as.push_back(a); bs.push_back(b); cs.push_back(0); expect.push_back(one_f32);
  }
  // 3. mismatched k pairs must give 0
  for (int k1 = 0; k1 < K; ++k1)
    for (int k2 = 0; k2 < K; ++k2) {
      if (k1 == k2) continue;
      auto a = zeros(), b = zeros(); a[k1] = one_elem; b[k2] = one_elem;
      as.push_back(a); bs.push_back(b); cs.push_back(0); expect.push_back(0u);
    }
  // 4. all ones: K
  {
    std::vector<uint32_t> a(K, one_elem), b(K, one_elem);
    as.push_back(a); bs.push_back(b); cs.push_back(0);
    expect.push_back(K == 32 ? 0x42000000u : 0x41000000u);
  }
  const uint32_t n = (uint32_t)as.size();
  std::vector<uint8_t> rec((size_t)n * rec_bytes);
  for (uint32_t i = 0; i < n; ++i) {
    uint8_t* p = rec.data() + (size_t)i * rec_bytes;
    for (int k = 0; k < K; ++k) {
      if (ci.esize == 1) { p[k] = (uint8_t)as[i][k]; p[K + k] = (uint8_t)bs[i][k]; }
      else {
        p[2 * k] = as[i][k] & 0xFF; p[2 * k + 1] = as[i][k] >> 8;
        p[2 * K + 2 * k] = bs[i][k] & 0xFF; p[2 * K + 2 * k + 1] = bs[i][k] >> 8;
      }
    }
    std::memcpy(p + 2 * K * ci.esize, &cs[i], 4);
  }
  auto got = run_batch(cfg, rec, n);
  int bad = 0;
  for (uint32_t i = 0; i < n; ++i)
    if (got[i] != expect[i]) {
      if (bad < 10) std::fprintf(stderr, "selftest FAIL case %u: got %08x want %08x\n", i, got[i], expect[i]);
      ++bad;
    }
  if (got.size() && got[0] == 0xFFFFFFFFu) std::fprintf(stderr, "this build compiled the instruction out (wrong -arch / NO_FP8)\n");
  if (bad) std::printf("selftest FAILED (%d of %u cases)\n", bad, n);
  else std::printf("selftest passed (%u cases)\n", n);
  return bad ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "--info") {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    std::printf("%s,sm_%d%d,driver=%d,runtime=%d\n", p.name, p.major, p.minor, drv, rt);
    return 0;
  }
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <e4m3_k32|f16_k8> (--selftest | in.bin out.bin)\n", argv[0]);
    return 64;
  }
  const std::string name = argv[1];
  int cfg;
  if (name == "e4m3_k32") cfg = E4M3_K32;
  else if (name == "f16_k8") cfg = F16_K8;
  else { std::fprintf(stderr, "unknown config %s\n", argv[1]); return 64; }
  if (std::string(argv[2]) == "--selftest") return selftest(cfg);
  if (argc < 4) return 64;
  FILE* f = std::fopen(argv[2], "rb");
  if (!f) { std::perror("open in"); return 66; }
  uint32_t n = 0;
  if (std::fread(&n, 4, 1, f) != 1) return 65;
  const CfgInfo ci = info_of(cfg);
  const uint32_t rec_bytes = 2 * ci.K * ci.esize + 4;
  std::vector<uint8_t> rec((size_t)n * rec_bytes);
  if (n && std::fread(rec.data(), 1, rec.size(), f) != rec.size()) { std::fprintf(stderr, "short read\n"); return 65; }
  std::fclose(f);
  auto out = run_batch(cfg, rec, n);
  FILE* g = std::fopen(argv[3], "wb");
  if (!g) { std::perror("open out"); return 66; }
  if (n) std::fwrite(out.data(), 4, n, g);
  std::fclose(g);
  return 0;
}
