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
// Pass 1 (EventRowQueueKernel): one thread per presynaptic row scans the batch,
// and every active row appends one (row, chunk, batch mask) item per
// `kEventChunk` edges, with one atomic per warp. Chunking spreads long rows
// (recurrent rows reach ~3k edges; the 100 BKG rows ~8k each) over many blocks.
//
// Pass 2 (EventWeightGradKernel): one block per item; each thread owns
// `kEventChunk / kEventThreads` edges and rebuilds their projections in FP32
// from `current_grad` and the basis for just the row's active samples, in
// ascending batch order. Up to 32 samples those come from the item's mask; a
// larger batch is compacted through shared memory, 256 samples per pass. Each
// edge has one writer and a fixed summation order, so the result is
// deterministic and needs no atomics.
//
// Needs `V1_EDGE_INDEX`, which the including file defines to match its weight
// layout. The resource library includes both operator sources into one
// translation unit, the external one inside a namespace, so this header is
// guarded and only depends on the tensorflow types both sources already use.
#pragma once

constexpr int kEventThreads = 256;
constexpr int kEventChunk = 4 * kEventThreads;
// Fixed consumer grid (188 SMs x 24 resident blocks on the RTX PRO 6000). Each
// block strides the queue and reads its length on the device, so no launch
// waits for the host to learn a count.
constexpr int kEventBlocks = 4512;

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

template <typename T>
__global__ __launch_bounds__(kEventThreads) void EventRowQueueKernel(
    int64_t n_pre, int64_t batch, const T* activity, const uint32* row_splits,
    unsigned int* queue, unsigned int* queue_count) {
  const int lane = threadIdx.x & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned int chunks = 0;
  unsigned int mask = 0;
  if (row < n_pre) {
    bool active = false;
    if (batch <= 32) {
      // The whole batch in one unrolled, branch-free pass: every load is
      // independent, and the mask it builds is what pass 2 walks.
#pragma unroll
      for (int sample = 0; sample < 32; ++sample) {
        if (sample < batch) {
          mask |= static_cast<unsigned int>(
                      static_cast<float>(activity[sample * n_pre + row]) != 0.0f)
                  << sample;
        }
      }
      active = mask != 0;
    } else {
      // Eight independent loads per step keep the scan throughput-bound; an
      // early exit after every sample would chain each load on the previous one.
      constexpr int kGroup = 8;
      for (int64_t first = 0; first < batch && !active; first += kGroup) {
#pragma unroll
        for (int k = 0; k < kGroup; ++k) {
          const int64_t sample = first + k;
          if (sample < batch) {
            active |= static_cast<float>(activity[sample * n_pre + row]) != 0.0f;
          }
        }
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
        ::make_uint4(static_cast<unsigned int>(row), chunk, mask, 0u);
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

// One edge's projection for one sample: `grad` is that sample's current_grad
// row for the edge's postsynaptic neuron. The four-column basis row `bv` is
// loaded once per edge by the caller.
template <typename T, int kBasis>
__device__ __forceinline__ float EdgeProjection(const T* grad, const float* type_basis,
                                                const ::float4& bv, int n_basis) {
  if constexpr (kBasis == 4) {
    return bv.x * static_cast<float>(grad[0]) + bv.y * static_cast<float>(grad[1]) +
           bv.z * static_cast<float>(grad[2]) + bv.w * static_cast<float>(grad[3]);
  } else {
    return EventProjection(grad, type_basis, n_basis);
  }
}

template <typename T, int kBasis>
__global__ __launch_bounds__(kEventThreads) void EventWeightGradKernel(
    int64_t n_pre, int n_post, int n_basis, int64_t batch, const T* activity,
    const T* current_grad, const float* basis, const uint32* post_ids,
    const uint8* synapse_types, const uint32* row_splits, const uint32* edge_ids,
    const unsigned int* queue, const unsigned int* queue_count, float* weight_grad) {
  constexpr int kEdgesPerThread = kEventChunk / kEventThreads;
  constexpr int kWarps = kEventThreads / 32;
  __shared__ int active_sample[kEventThreads];
  __shared__ float active_value[kEventThreads];
  __shared__ int warp_active[kWarps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t sample_stride = static_cast<int64_t>(n_post) * n_basis;
  const unsigned int total = *queue_count;
  for (unsigned int item = blockIdx.x; item < total; item += gridDim.x) {
    const ::uint4 entry = reinterpret_cast<const ::uint4*>(queue)[item];
    const int64_t pre = entry.x;
    const uint32 start = row_splits[pre] + entry.y * kEventChunk;
    const uint32 end = min(static_cast<uint32>(start + kEventChunk), row_splits[pre + 1]);
    float sum[kEdgesPerThread] = {};
    // Each pass covers up to kEventThreads samples. Up to 32 the item's mask
    // lists them; beyond that they are compacted through shared memory.
    for (int64_t first = 0; first < batch; first += kEventThreads) {
      int count = 0;
      if (batch > 32) {
        const int64_t sample = first + threadIdx.x;
        const float value =
            sample < batch ? static_cast<float>(activity[sample * n_pre + pre]) : 0.0f;
        const unsigned int ballot = __ballot_sync(0xffffffffu, value != 0.0f);
        if (lane == 0) warp_active[warp] = __popc(ballot);
        __syncthreads();
        int offset = 0;
#pragma unroll
        for (int other = 0; other < kWarps; ++other) {
          offset += other < warp ? warp_active[other] : 0;
          count += warp_active[other];
        }
        if (value != 0.0f) {
          const int slot = offset + __popc(ballot & ((1u << lane) - 1));
          active_sample[slot] = static_cast<int>(sample);
          active_value[slot] = value;
        }
        __syncthreads();
      }
#pragma unroll
      for (int k = 0; k < kEdgesPerThread; ++k) {
        const uint32 csr = start + threadIdx.x + k * kEventThreads;
        if (csr >= end) continue;
        const float* type_basis = basis + static_cast<int64_t>(synapse_types[csr]) * n_basis;
        const ::float4 bv = kBasis == 4 ? *reinterpret_cast<const ::float4*>(type_basis)
                                        : ::make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const T* grad_row = current_grad + static_cast<int64_t>(post_ids[csr]) * n_basis;
        if (batch <= 32) {
          for (unsigned int bits = entry.z; bits; bits &= bits - 1) {
            const int sample = __ffs(bits) - 1;
            sum[k] += static_cast<float>(activity[sample * n_pre + pre]) *
                      EdgeProjection<T, kBasis>(grad_row + sample * sample_stride, type_basis,
                                                bv, n_basis);
          }
        } else {
          for (int index = 0; index < count; ++index) {
            sum[k] += active_value[index] *
                      EdgeProjection<T, kBasis>(grad_row + active_sample[index] * sample_stride,
                                                type_basis, bv, n_basis);
          }
        }
      }
      if (batch > 32) __syncthreads();   // the next pass overwrites the shared list
    }
#pragma unroll
    for (int k = 0; k < kEdgesPerThread; ++k) {
      const uint32 csr = start + threadIdx.x + k * kEventThreads;
      if (csr < end) weight_grad[V1_EDGE_INDEX(csr)] += sum[k];
    }
  }
}

// Writes the event-driven weight gradient into `weight_grad`, which it clears
// first, or with `accumulate` adds it to what `weight_grad` already holds.
// `activity` is [batch, n_pre]; `current_grad` is [batch * n_post, n_basis].
template <typename T, int kBasis>
Status LaunchEventWeightGrad(OpKernelContext* context, const Tensor& activity,
                             const Tensor& current_grad, const Tensor& basis,
                             const Tensor& post_ids, const Tensor& synapse_types,
                             const Tensor& row_splits, const Tensor& edge_ids,
                             int n_post, Tensor* weight_grad, bool accumulate = false) {
  auto device = context->eigen_device<GPUDevice>();
  if (!accumulate)
    cudaMemsetAsync(weight_grad->flat<float>().data(), 0,
                    weight_grad->NumElements() * sizeof(float), device.stream());
  const int64_t batch = activity.dim_size(0);
  const int64_t n_pre = activity.dim_size(1);
  if (batch == 0 || n_pre == 0 || weight_grad->NumElements() == 0) return OkStatus();
  // Every active row yields ceil(edges / kEventChunk) items, so the queue holds
  // at most one item per row plus one per full chunk.
  const int64_t capacity = n_pre + post_ids.NumElements() / kEventChunk + 1;
  Tensor queue_tensor;
  unsigned int* queue;
  TF_RETURN_IF_ERROR(AllocateQueueWords(context, 4 * capacity + 1, &queue_tensor, &queue));
  unsigned int* queue_count = queue + 4 * capacity;
  cudaMemsetAsync(queue_count, 0, sizeof(unsigned int), device.stream());
  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      EventRowQueueKernel<T>, static_cast<int>((n_pre + kEventThreads - 1) / kEventThreads),
      kEventThreads, 0, device.stream(), n_pre, batch, activity.flat<T>().data(),
      row_splits.flat<uint32>().data(), queue, queue_count));
  return GpuLaunchKernel(
      EventWeightGradKernel<T, kBasis>, kEventBlocks, kEventThreads, 0, device.stream(),
      n_pre, n_post, static_cast<int>(basis.dim_size(1)), batch,
      activity.flat<T>().data(), current_grad.flat<T>().data(),
      basis.flat<float>().data(), post_ids.flat<uint32>().data(),
      synapse_types.flat<uint8>().data(), row_splits.flat<uint32>().data(),
      edge_ids.flat<uint32>().data(), queue, queue_count,
      weight_grad->flat<float>().data());
}
