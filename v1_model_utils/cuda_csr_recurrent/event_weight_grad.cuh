// Event-driven weight gradient shared by the recurrent and external operators:
//
//   dW[e] = sum_b activity[b, pre(e)] * sum_r current_grad[b, post(e), r] * basis[type(e), r]
//
// over the presynaptic rows where some batch sample is active. Activity is
// sparse, so a dense sweep of every edge would spend almost all of its work on
// rows that contribute exactly zero. Inactive rows are skipped, so a non-finite
// upstream gradient no longer reaches the weight gradient of edges whose row
// never fired; everything else is exact in finite arithmetic.
//
// Pass 1 (EventRowQueueKernel): one thread per presynaptic row scans the batch
// into a bit mask of its active samples, and every active row appends one
// (row, chunk, mask) item per `kEventChunk` edges, with one atomic per warp.
// Chunking spreads long rows (recurrent rows reach ~3k edges; the 100 BKG rows
// ~8k each) over many warps. Up to 64 samples the mask travels in the item;
// a larger batch stores it once per row, as ceil(batch / 32) words.
//
// Pass 2 (EventWeightGradWarpKernel) gives each item one warp, so a block works
// on eight items at once, and rebuilds each edge's projections in FP32 from
// `current_grad` and the basis for just the row's active samples, in ascending
// batch order. Each edge has one writer and a fixed summation order, so the
// result is deterministic and needs no atomics.
//
// Needs `V1_EDGE_INDEX`, which the including file defines to match its weight
// layout. The resource library includes both operator sources into one
// translation unit, the external one inside a namespace, so this header is
// guarded and only depends on the tensorflow types both sources already use.
#pragma once

// Exact n / divisor for n < 2^31 and 1 <= divisor < 2^31 with one multiply-high
// (the round-up method of Granlund and Montgomery), where a hardware division
// would cost ~20 instructions per element of a sweep.
struct FastDivider {
  unsigned int divisor;
  unsigned int magic;
  unsigned int shift;

  static FastDivider Make(unsigned int divisor) {
    unsigned int shift = 0;
    while ((1ull << shift) < divisor) ++shift;
    const unsigned long long magic =
        ((1ull << 32) * ((1ull << shift) - divisor)) / divisor + 1;
    return FastDivider{divisor, static_cast<unsigned int>(magic), shift};
  }

  __device__ __forceinline__ unsigned int Div(unsigned int n) const {
    return (__umulhi(n, magic) + n) >> shift;
  }
};

// A [batch, n_pre] spike (or activity) matrix held as `count` tensors of
// [batch, width], n_pre = count * width: presynaptic row `pre` is column
// pre % width of tensor pre / width. The recurrent spike history is kept this
// way, one tensor per delay slot, newest first, so that shifting it copies
// nothing (models.V1Column); every other operand is one tensor, count == 1,
// and addresses exactly as the plain matrix. P is `const T*` for inputs and
// `T*` for gradients.
constexpr int kMaxSpikeSlots = 8;

template <typename P>
struct SpikeSlots {
  P slot[kMaxSpikeSlots];
  int count;
  unsigned int width;
  unsigned int n_pre;
  FastDivider by_width;

  // Sample 0 of row `pre`; sample b is b * width elements further.
  __device__ __forceinline__ P Row(unsigned int pre) const {
    if (count == 1) return slot[0] + pre;
    const unsigned int index = by_width.Div(pre);
    P base = slot[0];
    // Constant indices keep the pointers in the parameter bank.
#pragma unroll
    for (int k = 1; k < kMaxSpikeSlots; ++k) {
      if (k == static_cast<int>(index)) base = slot[k];
    }
    return base + (pre - index * width);
  }

  __device__ __forceinline__ decltype(*P()) At(int64_t sample, unsigned int pre) const {
    return Row(pre)[sample * width];
  }
};

// The host side: the slots plus the batch they share.
template <typename P>
struct SpikeMatrix {
  SpikeSlots<P> slots;
  int64_t batch;
  int64_t n_pre() const { return slots.n_pre; }
  int64_t NumElements() const { return batch * slots.n_pre; }
};

// One [batch, n_pre] tensor as a single slot.
template <typename T, typename P = const T*>
SpikeMatrix<P> SingleSlot(P data, int64_t batch, int64_t n_pre) {
  SpikeMatrix<P> matrix{};
  matrix.slots.slot[0] = data;
  matrix.slots.count = 1;
  matrix.slots.width = static_cast<unsigned int>(n_pre);
  matrix.slots.n_pre = static_cast<unsigned int>(n_pre);
  matrix.slots.by_width = FastDivider::Make(std::max<unsigned int>(1, matrix.slots.width));
  matrix.batch = batch;
  return matrix;
}

// Equal-shape rank-two tensors as the slots of one matrix. Every width and
// n_pre must fit a 31-bit index, which is what FastDivider needs.
template <typename T, typename Tensors>
Status ReadSpikeSlots(const Tensors& tensors, SpikeMatrix<const T*>* matrix) {
  const int count = tensors.size();
  if (count < 1 || count > kMaxSpikeSlots) {
    return errors::InvalidArgument("between 1 and ", kMaxSpikeSlots, " spike slots, got ", count);
  }
  const Tensor& first = tensors[0];
  if (first.dims() != 2) return errors::InvalidArgument("spikes must be rank two");
  const int64_t n_pre = first.dim_size(1) * count;
  if (n_pre >= (int64_t{1} << 31)) {
    return errors::InvalidArgument("spike slots are too wide: ", n_pre, " rows");
  }
  *matrix = SingleSlot<T>(first.flat<T>().data(), first.dim_size(0), n_pre);
  matrix->slots.count = count;
  matrix->slots.width = static_cast<unsigned int>(first.dim_size(1));
  matrix->slots.by_width = FastDivider::Make(std::max<unsigned int>(1, matrix->slots.width));
  for (int k = 0; k < count; ++k) {
    if (tensors[k].shape() != first.shape()) {
      return errors::InvalidArgument("every spike slot must have the shape of the first");
    }
    matrix->slots.slot[k] = tensors[k].template flat<T>().data();
  }
  return OkStatus();
}

constexpr int kEventThreads = 256;
constexpr int kEventChunk = kEventThreads;
// Fixed consumer grid (188 SMs x 24 blocks on the RTX PRO 6000). Fewer are
// resident at once, so later blocks start as earlier ones finish; for the event
// weight gradient this times the same as 2256 to 18048 blocks, and faster than
// a grid of only the resident blocks. Each block strides the queue and reads its
// length on the device, so no launch waits for the host to learn a count.
constexpr int kEventBlocks = 4512;
// Batches up to this size carry their active samples as a bit mask in the queue
// item (two words); a larger batch keeps one mask per row in the queue scratch.
constexpr int kEventMaskSamples = 64;

// `words` of scratch for a device-built queue. Allocated as int32, not uint32:
// the resource library compiles the operator sources with `uint32` redefined as
// `int32`, so a DT_UINT32 tensor read through `flat<uint32>()` would fail its
// dtype check there.
inline Status AllocateQueueWords(OpKernelContext* context, int64_t words, Tensor* tensor,
                                 unsigned int** data) {
  TF_RETURN_IF_ERROR(context->allocate_temp(DT_INT32, TensorShape({words}), tensor));
  *data = reinterpret_cast<unsigned int*>(tensor->flat<int32>().data());
  return OkStatus();
}

// `row_masks` is null when the batch fits the item mask; otherwise word w of
// row `pre`'s mask, samples 32w to 32w + 31, is row_masks[pre * mask_words + w],
// so pass 2 reads a row's mask (up to 256 samples) from one 32-byte sector.
template <typename T>
__global__ __launch_bounds__(kEventThreads) void EventRowQueueKernel(
    int64_t n_pre, int64_t batch, SpikeSlots<const T*> slots, const uint32* row_splits,
    unsigned int* row_masks, int mask_words, unsigned int* queue, unsigned int* queue_count) {
  const int lane = threadIdx.x & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned int chunks = 0;
  unsigned int mask = 0;
  unsigned int mask_high = 0;
  if (row < n_pre) {
    // activity[sample * stride] is (sample, row).
    const T* activity = slots.Row(static_cast<unsigned int>(row));
    const int64_t stride = slots.width;
    bool active = false;
    if (batch <= kEventMaskSamples) {
      // The whole batch in one unrolled, branch-free pass: every load is
      // independent, and the 64-bit mask it builds is what pass 2 walks.
#pragma unroll
      for (int sample = 0; sample < kEventMaskSamples; ++sample) {
        if (sample < batch) {
          const unsigned int bit =
              static_cast<float>(activity[sample * stride]) != 0.0f;
          if (sample < 32) {
            mask |= bit << sample;
          } else {
            mask_high |= bit << (sample - 32);
          }
        }
      }
      active = (mask | mask_high) != 0;
    } else {
      // Every sample, 32 independent loads per mask word. An early exit at the
      // first active sample would save almost no traffic: a warp's rows share
      // each sample's sectors, so they are read while any of its rows scans.
      for (int word = 0; word < mask_words; ++word) {
        unsigned int bits = 0;
#pragma unroll
        for (int bit = 0; bit < 32; ++bit) {
          const int64_t sample = word * 32 + bit;
          if (sample < batch) {
            bits |= static_cast<unsigned int>(static_cast<float>(activity[sample * stride]) !=
                                              0.0f) << bit;
          }
        }
        row_masks[row * mask_words + word] = bits;
        active |= bits != 0;
      }
    }
    if (active) {
      const unsigned int edges =
          static_cast<unsigned int>(row_splits[row + 1] - row_splits[row]);
      chunks = (edges + kEventChunk - 1) / kEventChunk;
    }
  }
  // Warp-inclusive scan of the chunk counts, then one atomic per warp.
  unsigned int inclusive = chunks;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const unsigned int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  const unsigned int total = __shfl_sync(0xffffffffu, inclusive, 31);
  if (total == 0) return;
  unsigned int base = 0;
  if (lane == 31) base = atomicAdd(queue_count, total);
  base = __shfl_sync(0xffffffffu, base, 31) + inclusive - chunks;
  for (unsigned int chunk = 0; chunk < chunks; ++chunk) {
    reinterpret_cast<::uint4*>(queue)[base + chunk] =
        ::make_uint4(static_cast<unsigned int>(row), chunk, mask, mask_high);
  }
}

// sum_r grad[r] * basis[r] over a synapse type's basis row.
template <typename T>
__device__ __forceinline__ float EventProjection(const T* grad, const float* basis,
                                                 int n_basis) {
  float result = 0.0f;
  for (int receptor = 0; receptor < n_basis; ++receptor) {
    result += basis[receptor] * static_cast<float>(grad[receptor]);
  }
  return result;
}

// A four-receptor current_grad row in one aligned vector load (8 bytes of
// FP16, 16 of FP32) instead of four scalar loads.
template <typename T>
__device__ __forceinline__ void LoadEventGrad4(const T* grad, float* out) {
  if constexpr (sizeof(T) == 2) {
    const ::uint2 raw = *reinterpret_cast<const ::uint2*>(grad);
    const T* packed = reinterpret_cast<const T*>(&raw);
#pragma unroll
    for (int receptor = 0; receptor < 4; ++receptor) out[receptor] = static_cast<float>(packed[receptor]);
  } else {
    const ::float4 raw = *reinterpret_cast<const ::float4*>(grad);
    out[0] = raw.x; out[1] = raw.y; out[2] = raw.z; out[3] = raw.w;
  }
}

// One edge's projection for one sample: `grad` is that sample's current_grad
// row for the edge's postsynaptic neuron. The four-column basis row `bv` is
// loaded once per edge by the caller.
template <typename T, int kBasis>
__device__ __forceinline__ float EdgeProjection(const T* grad, const float* type_basis,
                                                const ::float4& bv, int n_basis) {
  if constexpr (kBasis == 4) {
    float g[4];
    LoadEventGrad4<T>(grad, g);
    return bv.x * g[0] + bv.y * g[1] + bv.z * g[2] + bv.w * g[3];
  } else {
    return EventProjection(grad, type_basis, n_basis);
  }
}

// Pass 2: one warp per queue item, lane l owning edges l, l + 32, ... of the
// item's chunk. One block per item left most of its threads idle on the typical
// ~60-edge row and serialized each item's chain of dependent loads (queue
// entry, row extent, edge, gradient, weight gradient); eight items per block
// overlap those chains. Each lane loads its edges' current weight gradient up
// front, and walks the samples outermost so that the gathers of its edges are
// independent. Up to kEventMaskSamples the samples come from the item's 64-bit
// mask; with kRowMasks from the row's mask words, one word per lane and 32 words
// per pass, the warp visiting the non-zero words in order. Per edge: sum +=
// activity * EdgeProjection over the active samples in ascending order, then
// one write of old + sum.
template <typename T, int kBasis, bool kRowMasks>
__global__ __launch_bounds__(kEventThreads) void EventWeightGradWarpKernel(
    int n_post, int n_basis, SpikeSlots<const T*> slots, const T* current_grad,
    const float* basis, const uint32* post_ids, const uint8* synapse_types,
    const uint32* row_splits, const uint32* edge_ids, const unsigned int* row_masks,
    int mask_words, const unsigned int* queue, const unsigned int* queue_count,
    float* weight_grad) {
  constexpr int kLaneEdges = kEventChunk / 32;
  constexpr int kWarps = kEventThreads / 32;
  const int lane = threadIdx.x & 31;
  const int64_t sample_stride = static_cast<int64_t>(n_post) * n_basis;
  const unsigned int total = *queue_count;
  for (unsigned int item = blockIdx.x * kWarps + (threadIdx.x >> 5); item < total;
       item += gridDim.x * kWarps) {
    const ::uint4 entry = reinterpret_cast<const ::uint4*>(queue)[item];
    // activity[sample * stride] is (sample, pre).
    const T* activity = slots.Row(entry.x);
    const int64_t stride = slots.width;
    const uint32 chunk_start = row_splits[entry.x] + entry.y * kEventChunk;
    const uint32 end = min(static_cast<uint32>(chunk_start + kEventChunk),
                           row_splits[entry.x + 1]);
    const uint32 start = chunk_start + lane;
    uint32 post[kLaneEdges];
    uint32 type[kLaneEdges];
    float old[kLaneEdges];
    float sum[kLaneEdges];
#pragma unroll
    for (int k = 0; k < kLaneEdges; ++k) {
      const uint32 csr = start + k * 32;
      const bool own = csr < end;
      post[k] = own ? post_ids[csr] : 0;
      type[k] = own ? synapse_types[csr] : 0;
      old[k] = own ? weight_grad[V1_EDGE_INDEX(csr)] : 0.0f;
      sum[k] = 0.0f;
    }
    const auto add_sample = [&](int sample) {
      const float value = static_cast<float>(activity[sample * stride]);
#pragma unroll
      for (int k = 0; k < kLaneEdges; ++k) {
        if (start + k * 32 < end) {
          const float* type_basis = basis + static_cast<int64_t>(type[k]) * n_basis;
          const ::float4 bv = kBasis == 4 ? *reinterpret_cast<const ::float4*>(type_basis)
                                          : ::make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const T* grad_row = current_grad + static_cast<int64_t>(post[k]) * n_basis;
          sum[k] += value * EdgeProjection<T, kBasis>(grad_row + sample * sample_stride,
                                                      type_basis, bv, n_basis);
        }
      }
    };
    if constexpr (kRowMasks) {
      for (int first_word = 0; first_word < mask_words; first_word += 32) {
        const int own_word = first_word + lane;
        const unsigned int word =
            own_word < mask_words
                ? row_masks[static_cast<int64_t>(entry.x) * mask_words + own_word]
                : 0u;
        for (unsigned int words = __ballot_sync(0xffffffffu, word != 0); words;
             words &= words - 1) {
          const int index = __ffs(words) - 1;
          const int first_sample = (first_word + index) * 32;
          for (unsigned int bits = __shfl_sync(0xffffffffu, word, index); bits;
               bits &= bits - 1) {
            add_sample(first_sample + __ffs(bits) - 1);
          }
        }
      }
    } else {
      const unsigned long long item_mask =
          (static_cast<unsigned long long>(entry.w) << 32) | entry.z;
      for (unsigned long long bits = item_mask; bits; bits &= bits - 1) {
        add_sample(__ffsll(bits) - 1);
      }
    }
#pragma unroll
    for (int k = 0; k < kLaneEdges; ++k) {
      const uint32 csr = start + k * 32;
      if (csr < end) weight_grad[V1_EDGE_INDEX(csr)] = old[k] + sum[k];
    }
  }
}

// Writes the event-driven weight gradient into `weight_grad`, which it clears
// first, or with `accumulate` adds it to what `weight_grad` already holds.
// `activity` is [batch, n_pre]; `current_grad` is [batch * n_post, n_basis].
template <typename T, int kBasis>
Status LaunchEventWeightGrad(OpKernelContext* context, const SpikeMatrix<const T*>& activity,
                             const Tensor& current_grad, const Tensor& basis,
                             const Tensor& post_ids, const Tensor& synapse_types,
                             const Tensor& row_splits, const Tensor& edge_ids,
                             int n_post, Tensor* weight_grad, bool accumulate = false) {
  auto device = context->eigen_device<GPUDevice>();
  if (!accumulate)
    cudaMemsetAsync(weight_grad->flat<float>().data(), 0,
                    weight_grad->NumElements() * sizeof(float), device.stream());
  const int64_t batch = activity.batch;
  const int64_t n_pre = activity.n_pre();
  if (batch == 0 || n_pre == 0 || weight_grad->NumElements() == 0) return OkStatus();
  // Every active row yields ceil(edges / kEventChunk) items, so the queue holds
  // at most one item per row plus one per full chunk. Above kEventMaskSamples
  // the row masks follow the queue, and the count comes last.
  const int64_t capacity = n_pre + post_ids.NumElements() / kEventChunk + 1;
  const bool item_masks = batch <= kEventMaskSamples;
  const int mask_words = item_masks ? 2 : static_cast<int>((batch + 31) / 32);
  const int64_t row_mask_words = item_masks ? 0 : n_pre * mask_words;
  Tensor queue_tensor;
  unsigned int* queue;
  TF_RETURN_IF_ERROR(AllocateQueueWords(context, 4 * capacity + row_mask_words + 1,
                                        &queue_tensor, &queue));
  unsigned int* row_masks = item_masks ? nullptr : queue + 4 * capacity;
  unsigned int* queue_count = queue + 4 * capacity + row_mask_words;
  cudaMemsetAsync(queue_count, 0, sizeof(unsigned int), device.stream());
  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      EventRowQueueKernel<T>, static_cast<int>((n_pre + kEventThreads - 1) / kEventThreads),
      kEventThreads, 0, device.stream(), n_pre, batch, activity.slots,
      row_splits.flat<uint32>().data(), row_masks, mask_words, queue, queue_count));
  const auto pass2 = item_masks ? EventWeightGradWarpKernel<T, kBasis, false>
                                : EventWeightGradWarpKernel<T, kBasis, true>;
  return GpuLaunchKernel(
      pass2, kEventBlocks, kEventThreads, 0, device.stream(),
      n_post, static_cast<int>(basis.dim_size(1)), activity.slots,
      current_grad.flat<T>().data(), basis.flat<float>().data(),
      post_ids.flat<uint32>().data(), synapse_types.flat<uint8>().data(),
      row_splits.flat<uint32>().data(), edge_ids.flat<uint32>().data(), row_masks, mask_words,
      queue, queue_count, weight_grad->flat<float>().data());
}

// The same for one [batch, n_pre] activity tensor.
template <typename T, int kBasis>
Status LaunchEventWeightGrad(OpKernelContext* context, const Tensor& activity,
                             const Tensor& current_grad, const Tensor& basis,
                             const Tensor& post_ids, const Tensor& synapse_types,
                             const Tensor& row_splits, const Tensor& edge_ids,
                             int n_post, Tensor* weight_grad, bool accumulate = false) {
  return LaunchEventWeightGrad<T, kBasis>(
      context, SingleSlot<T>(activity.flat<T>().data(), activity.dim_size(0), activity.dim_size(1)),
      current_grad, basis, post_ids, synapse_types, row_splits, edge_ids, n_post, weight_grad,
      accumulate);
}
