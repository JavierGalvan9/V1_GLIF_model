// TQ-pack: the frame store's error-bounded temporal codec for float32 frames
// (included by frame_store_ops.cu.cc; no library dependency).
//
// Value x of frame s is quantized to q = rint(x * 2^E) (round half to even;
// x * 2^E is exact, so |error| <= 2^-(E+1) and the reconstruction q * 2^-E is
// exactly rint(x * 2^E) * 2^-E). It is predicted from the two *later* frames
// (lagged encoding, decoding in reverse time, the order the reverse pass takes
// the frames): p = 2 q(s+1) - q(s+2) (order kOrder = 2; the forward's last
// two frames, and frames drained before their references are recorded, use
// the lower orders 1: p = q(s+1) and 0: p = 0),
// where q(.) of a reference is the same quantizer applied to its value (0 for
// |value| >= 128 or NaN). Quantizing a reference's raw value or its
// reconstruction gives the same integer (a reconstruction is on the grid, and
// raw blocks below are exact), so the encoder predicts from the raw frames and
// the decoder from the reconstructed ones without drift.
// The residual r = q - p is ZigZag-coded; each block of 32 consecutive values
// (one warp) stores its bit width w in one byte and w bit-plane words. A block
// holding a non-finite value or |x| >= 127 is stored raw (code kRawCode: 32
// words of fp32 bits), so every value is representable and the integers stay
// within int32 for E <= 21. E = kExponent = 16 (grid 2^-16, |error| <= 2^-17
// ~= 7.6e-6).
// One encoded frame: [tile offsets: tiles x u32][widths: blocks x u8, padded to
// 16 B][payload: u32 words], written by one kernel with single-pass compaction
// (decoupled look-back over 4096-value tiles); its length is written on the
// device. Frames go into a ring arena: a one-block prologue kernel places the
// frame at the ring's device-side cursor (wrapping to the start when a
// worst-case frame does not fit before the end) if the worst-case frame fits
// below the host's tail, and otherwise marks it "ring full" (length 0: the
// encode kernel does nothing and the frame stays raw).

namespace tq {

constexpr int kWarps = 8;
constexpr int kBlocksPerWarp = 16;
constexpr int kTileBlocks = kWarps * kBlocksPerWarp;  // 128 blocks = 4096 values
constexpr int kRawCode = 33;
constexpr int kOrder = 2;      // references per frame (at most)
constexpr int kExponent = 16;  // the grid 2^-kExponent
static_assert(kExponent <= 21, "|residual| < 2^30 for |x| < 128 needs E <= 21");
constexpr float kScale = 1 << kExponent;          // 2^E
constexpr float kDelta = 1.0f / (1 << kExponent);  // 2^-E
constexpr unsigned long long kFull = ~0ull;
constexpr size_t kAlign = 256;  // frames start at multiples of this in the ring

// Where the parts of an encoded frame of n values lie, and its largest size.
struct Layout {
  long long n = 0, blocks = 0, tiles = 0;
  size_t widths = 0, payload = 0, max_bytes = 0;
};

inline Layout MakeLayout(long long n) {
  Layout layout;
  layout.n = n;
  layout.blocks = (n + 31) / 32;
  layout.tiles = (layout.blocks + kTileBlocks - 1) / kTileBlocks;
  layout.widths = static_cast<size_t>(layout.tiles) * 4;
  layout.payload = (layout.widths + static_cast<size_t>(layout.blocks) + 15) / 16 * 16;
  layout.max_bytes = layout.payload + static_cast<size_t>(layout.blocks) * 32 * 4;
  return layout;
}

inline size_t AlignUp(size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

// The device-side state of a ring: its cursor (virtual byte position, the end
// of the latest encoded frame), the base chosen for the frame being encoded
// (kFull: ring full) and the tile counter of the look-back.
struct Scratch {
  unsigned long long cursor;
  unsigned long long base;
  unsigned int counter;
};

// Where an encode put its frame (read by the host once the encode ended).
struct Result {
  unsigned long long position;
  unsigned long long length;  // 0: ring full, not encoded
};

__device__ __forceinline__ int Quantize(float x, float scale) {
  return fabsf(x) < 128.0f ? __float2int_rn(x * scale) : 0;  // NaN -> 0
}

template <int ORDER>
__device__ __forceinline__ int Predict(const float* r1, const float* r2, long long i,
                                       float scale) {
  if (ORDER == 0) return 0;
  const int p1 = Quantize(r1[i], scale);
  if (ORDER == 1) return p1;
  return 2 * p1 - Quantize(r2[i], scale);
}

// Places the next frame at the cursor (or the ring's start) if a worst-case
// frame fits below tail + ring_bytes, and resets the look-back state.
__global__ void PlaceKernel(Scratch* scratch, unsigned long long* status, long long tiles,
                            unsigned long long tail, unsigned long long ring_bytes,
                            unsigned long long max_bytes, volatile Result* result) {
  if (threadIdx.x == 0) {
    unsigned long long position = scratch->cursor;
    if (position % ring_bytes + max_bytes > ring_bytes)
      position += ring_bytes - position % ring_bytes;
    const bool fits = position + max_bytes <= tail + ring_bytes;
    scratch->base = fits ? position : kFull;
    scratch->counter = 0;
    if (!fits) {
      result->position = position;
      result->length = 0;
      __threadfence_system();
    }
  }
  for (long long tile = threadIdx.x; tile < tiles; tile += blockDim.x) status[tile] = 0;
}

template <int ORDER>
__global__ void __launch_bounds__(kWarps * 32)
    EncodeKernel(const float* __restrict__ x, const float* __restrict__ r1,
                 const float* __restrict__ r2, long long n, float scale,
                 unsigned char* __restrict__ ring, unsigned long long ring_bytes,
                 Scratch* scratch, volatile unsigned long long* status, size_t widths_offset,
                 size_t payload_offset, volatile Result* result) {
  __shared__ unsigned int s_tile;
  __shared__ unsigned int s_warp[kWarps];
  __shared__ unsigned int s_base;
  const unsigned long long base = *reinterpret_cast<volatile unsigned long long*>(&scratch->base);
  if (base == kFull) return;
  unsigned char* frame = ring + base % ring_bytes;
  unsigned int* tile_offsets = reinterpret_cast<unsigned int*>(frame);
  unsigned char* widths = frame + widths_offset;
  unsigned int* payload = reinterpret_cast<unsigned int*>(frame + payload_offset);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) s_tile = atomicAdd(&scratch->counter, 1u);
  __syncthreads();
  const unsigned int tile = s_tile;
  const long long blocks = (n + 31) >> 5;
  unsigned int word[kBlocksPerWarp];
  int code[kBlocksPerWarp];
  unsigned int warp_words = 0;
#pragma unroll
  for (int b = 0; b < kBlocksPerWarp; ++b) {
    const long long block = static_cast<long long>(tile) * kTileBlocks + warp * kBlocksPerWarp + b;
    const long long i = block * 32 + lane;
    float v = 0.0f;
    int p = 0;
    if (block < blocks && i < n) {
      v = x[i];
      p = Predict<ORDER>(r1, r2, i, scale);
    }
    const bool raw = !(fabsf(v) < 127.0f);
    const int r = raw ? 0 : __float2int_rn(v * scale) - p;
    const unsigned int zz = (static_cast<unsigned int>(r) << 1) ^ static_cast<unsigned int>(r >> 31);
    const bool any_raw = __any_sync(0xffffffffu, raw);
    const unsigned int all = __reduce_or_sync(0xffffffffu, zz);
    const int w = any_raw ? kRawCode : 32 - __clz(all);
    code[b] = block < blocks ? w : 0;
    word[b] = any_raw ? __float_as_uint(v) : zz;
    warp_words += code[b] == kRawCode ? 32u : static_cast<unsigned int>(code[b]);
    if (lane == 0 && block < blocks) widths[block] = static_cast<unsigned char>(code[b]);
  }
  if (lane == 0) s_warp[warp] = warp_words;
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int aggregate = 0;
    for (int k = 0; k < kWarps; ++k) {
      const unsigned int words = s_warp[k];
      s_warp[k] = aggregate;
      aggregate += words;
    }
    unsigned long long prefix = 0;
    if (tile == 0) {
      status[0] = (2ull << 62) | aggregate;
    } else {
      status[tile] = (1ull << 62) | aggregate;
      long long j = static_cast<long long>(tile) - 1;
      for (;;) {
        unsigned long long s;
        do {
          s = status[j];
        } while ((s >> 62) == 0);
        prefix += s & ((1ull << 62) - 1);
        if ((s >> 62) == 2) break;
        --j;
      }
      __threadfence();
      status[tile] = (2ull << 62) | (prefix + aggregate);
    }
    tile_offsets[tile] = static_cast<unsigned int>(prefix);
    s_base = static_cast<unsigned int>(prefix);
    if (static_cast<long long>(tile + 1) * kTileBlocks >= blocks) {  // the last tile
      const unsigned long long length = payload_offset + 4ull * (prefix + aggregate);
      scratch->cursor = base + (length + kAlign - 1) / kAlign * kAlign;
      result->position = base;
      result->length = length;
      __threadfence_system();
    }
  }
  __syncthreads();
  unsigned int offset = s_base + s_warp[warp];
#pragma unroll
  for (int b = 0; b < kBlocksPerWarp; ++b) {
    const int w = code[b];
    if (w == kRawCode) {
      payload[offset + lane] = word[b];
      offset += 32;
    } else {
      for (int j = 0; j < w; ++j) {
        const unsigned int plane = __ballot_sync(0xffffffffu, (word[b] >> j) & 1u);
        if (lane == j) payload[offset + j] = plane;
      }
      offset += w;
    }
  }
}

template <int ORDER>
__global__ void __launch_bounds__(kWarps * 32)
    DecodeKernel(float* __restrict__ y, const float* __restrict__ r1,
                 const float* __restrict__ r2, long long n, float scale, float delta,
                 const unsigned char* __restrict__ frame, size_t widths_offset,
                 size_t payload_offset) {
  __shared__ unsigned int s_warp[kWarps];
  const unsigned int* tile_offsets = reinterpret_cast<const unsigned int*>(frame);
  const unsigned char* widths = frame + widths_offset;
  const unsigned int* payload = reinterpret_cast<const unsigned int*>(frame + payload_offset);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const unsigned int tile = blockIdx.x;
  const long long blocks = (n + 31) >> 5;
  const long long first = static_cast<long long>(tile) * kTileBlocks + warp * kBlocksPerWarp;
  int mine = 0;  // lanes 0..15 read one width each: the warp's word count
  if (lane < kBlocksPerWarp && first + lane < blocks) {
    mine = widths[first + lane];
    mine = mine == kRawCode ? 32 : mine;
  }
  const unsigned int total = __reduce_add_sync(0xffffffffu, static_cast<unsigned int>(mine));
  if (lane == 0) s_warp[warp] = total;
  __syncthreads();
  unsigned int offset = tile_offsets[tile];
  for (int k = 0; k < warp; ++k) offset += s_warp[k];
  for (int b = 0; b < kBlocksPerWarp; ++b) {
    const long long block = first + b;
    if (block >= blocks) break;
    const int w = widths[block];
    const long long i = block * 32 + lane;
    if (w == kRawCode) {
      const unsigned int bits = payload[offset + lane];
      if (i < n) y[i] = __uint_as_float(bits);
      offset += 32;
      continue;
    }
    const unsigned int word = lane < w ? payload[offset + lane] : 0u;
    unsigned int zz = 0;
    for (int j = 0; j < w; ++j) zz |= ((__shfl_sync(0xffffffffu, word, j) >> lane) & 1u) << j;
    offset += w;
    if (i < n) {
      const int r = static_cast<int>(zz >> 1) ^ -static_cast<int>(zz & 1u);
      y[i] = static_cast<float>(r + Predict<ORDER>(r1, r2, i, scale)) * delta;
    }
  }
}

// Encodes x (n values; references r1 = frame s+1, r2 = frame s+2 as the order
// needs) into the ring at the cursor, or marks the ring full; `result` gets
// the position and length.
inline cudaError_t Encode(const float* x, const float* r1, const float* r2, int order,
                          const Layout& layout, unsigned char* ring,
                          unsigned long long ring_bytes, unsigned long long tail,
                          Scratch* scratch, unsigned long long* status, Result* result,
                          cudaStream_t stream) {
  PlaceKernel<<<1, 256, 0, stream>>>(scratch, status, layout.tiles, tail, ring_bytes,
                                     layout.max_bytes, result);
  const dim3 grid(static_cast<unsigned int>(layout.tiles)), block(kWarps * 32);
  switch (order) {
    case 0:
      EncodeKernel<0><<<grid, block, 0, stream>>>(x, r1, r2, layout.n, kScale, ring, ring_bytes,
                                                 scratch, status, layout.widths,
                                                 layout.payload, result);
      break;
    case 1:
      EncodeKernel<1><<<grid, block, 0, stream>>>(x, r1, r2, layout.n, kScale, ring, ring_bytes,
                                                 scratch, status, layout.widths,
                                                 layout.payload, result);
      break;
    default:
      EncodeKernel<2><<<grid, block, 0, stream>>>(x, r1, r2, layout.n, kScale, ring, ring_bytes,
                                                 scratch, status, layout.widths,
                                                 layout.payload, result);
  }
  return cudaGetLastError();
}

// Decodes an encoded frame into y from the reconstructed references.
inline cudaError_t Decode(float* y, const float* r1, const float* r2, int order,
                          const Layout& layout, const unsigned char* frame, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(layout.tiles)), block(kWarps * 32);
  switch (order) {
    case 0:
      DecodeKernel<0><<<grid, block, 0, stream>>>(y, r1, r2, layout.n, kScale, kDelta, frame,
                                                 layout.widths, layout.payload);
      break;
    case 1:
      DecodeKernel<1><<<grid, block, 0, stream>>>(y, r1, r2, layout.n, kScale, kDelta, frame,
                                                 layout.widths, layout.payload);
      break;
    default:
      DecodeKernel<2><<<grid, block, 0, stream>>>(y, r1, r2, layout.n, kScale, kDelta, frame,
                                                 layout.widths, layout.payload);
  }
  return cudaGetLastError();
}

}  // namespace tq
