template <typename T, int kBasis, int kPairsPerTile>
__global__ void PreprojectGenericPairsKernel(
    int64_t batch_size, int n_post, int n_basis, int64_t n_pairs,
    const T* current_grad, const T* basis, const uint32* pair_posts,
    const uint8* pair_types, float* projected) {
  constexpr int kBatchTile = 32;
  constexpr int kElements = kPairsPerTile * kBatchTile;
  __shared__ float tile[kBatchTile][kPairsPerTile + 2];
  const int64_t pair_base = static_cast<int64_t>(blockIdx.x) * kPairsPerTile;
  const int64_t batch_base = static_cast<int64_t>(blockIdx.y) * kBatchTile;
  for (int index = threadIdx.x; index < kElements; index += blockDim.x) {
    const int column = index % kPairsPerTile;
    const int local_batch = index / kPairsPerTile;
    const int64_t pair = pair_base + column;
    const int64_t batch = batch_base + local_batch;
    float value = 0.0f;
    if (pair < n_pairs && batch < batch_size) {
      value = BasisProjection<T, kBasis>::Apply(
          current_grad + (batch * n_post + pair_posts[pair]) * n_basis,
          basis, pair_types[pair], n_basis);
    }
    tile[local_batch][column] = value;
  }
  __syncthreads();
  for (int index = threadIdx.x; index < kElements; index += blockDim.x) {
    const int local_batch = index % kBatchTile;
    const int column = index / kBatchTile;
    const int64_t pair = pair_base + column;
    const int64_t batch = batch_base + local_batch;
    if (pair < n_pairs && batch < batch_size) {
      projected[pair * batch_size + batch] = tile[local_batch][column];
    }
  }
}

template <typename T, typename W, int kWarps, int kTile>
__global__ __launch_bounds__(32 * kWarps) void GenericRecurrentWeightKernel(
    int64_t batch_size, int64_t n_pre, const T* spikes,
    const float* projected, const uint32* pair_ids, const uint32* edge_ids,
    const uint32* row_splits, const uint32* nonempty_rows, int64_t n_rows,
    float* weight_grad) {
  constexpr int kPack = 4;
  constexpr int kLanesPerEdge = 8;
  constexpr int kSlots = 4;
  constexpr int kPerSlot = kTile / kSlots;
  constexpr uint32 kGrain = kWarps * kTile;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int slot = lane / kLanesPerEdge;
  const int sub = lane % kLanesPerEdge;
  const int64_t row_id = blockIdx.x;
  if (row_id >= n_rows) return;
  const uint32 pre = nonempty_rows[row_id];
  const uint32 start = row_splits[pre];
  const uint32 end = row_splits[pre + 1];
  const int target = ReverseBits<3>(sub & (kPerSlot - 1));
  for (uint32 base = start + warp * kTile; base < end; base += kGrain) {
    const uint32 edge_lane = base + lane;
    const uint32 my_pair = edge_lane < end ? LoadEdgeIndex(pair_ids + edge_lane) : 0u;
    float partial[kPerSlot] = {};
    for (int64_t batch_base = 0; batch_base < batch_size; batch_base += 32) {
      float sample[kPack];
#pragma unroll
      for (int index = 0; index < kPack; ++index) {
        const int64_t batch = batch_base + sub * kPack + index;
        sample[index] = batch < batch_size
                            ? AsFloat(spikes[batch * n_pre + pre])
                            : 0.0f;
      }
#pragma unroll
      for (int step = 0; step < kPerSlot; ++step) {
        const int column = kSlots * step + slot;
        const uint32 pair = __shfl_sync(0xffffffff, my_pair, column);
        if (base + column < end) {
#pragma unroll
          for (int index = 0; index < kPack; ++index) {
            const int64_t batch = batch_base + sub * kPack + index;
            if (batch < batch_size) {
              partial[step] += projected[static_cast<int64_t>(pair) * batch_size + batch] * sample[index];
            }
          }
        }
      }
    }
    ButterflyReduce<kPerSlot / 2, 1>(partial, lane);
#pragma unroll
    for (int mask = kPerSlot; mask < kLanesPerEdge; mask <<= 1) {
      partial[0] += __shfl_xor_sync(0xffffffff, partial[0], mask);
    }
    const uint32 edge = base + kSlots * target + slot;
    if (sub < kPerSlot && edge < end) StoreEdgeGrad(weight_grad + V1_EDGE_INDEX(edge), partial[0]);
  }
}

template <typename T, typename W, int kWarps, int kTile>
__global__ __launch_bounds__(32 * kWarps) void GenericRecurrentSpikeKernel(
    int64_t batch_size, int64_t n_pre, const float* projected,
    const W* weights, const uint32* pair_ids, const uint32* edge_ids,
    const uint32* row_splits, const uint32* nonempty_rows, int64_t n_rows,
    const T* dampening, T* spike_grad) {
  constexpr int kPack = 4;
  constexpr int kLanesPerEdge = 8;
  constexpr int kSlots = 4;
  constexpr int kPerSlot = kTile / kSlots;
  constexpr uint32 kGrain = kWarps * kTile;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int slot = lane / kLanesPerEdge;
  const int sub = lane % kLanesPerEdge;
  const int64_t row_id = blockIdx.x;
  if (row_id >= n_rows) return;
  __shared__ float partials[kWarps][32];
  const uint32 pre = nonempty_rows[row_id];
  const uint32 start = row_splits[pre];
  const uint32 end = row_splits[pre + 1];
  const int64_t batch_base = static_cast<int64_t>(blockIdx.y) * 32;
  float grad[kPack] = {};
  for (uint32 base = start + warp * kTile; base < end; base += kGrain) {
    const uint32 edge_lane = base + lane;
    const bool own = edge_lane < end;
    const uint32 my_pair = own ? LoadEdgeIndex(pair_ids + edge_lane) : 0u;
    const float my_weight = own ? LoadEdgeWeight<W>(weights + V1_EDGE_INDEX(edge_lane)) : 0.0f;
#pragma unroll
    for (int step = 0; step < kPerSlot; ++step) {
      const int column = kSlots * step + slot;
      const uint32 pair = __shfl_sync(0xffffffff, my_pair, column);
      const float weight = __shfl_sync(0xffffffff, my_weight, column);
      if (base + column < end) {
#pragma unroll
        for (int index = 0; index < kPack; ++index) {
          const int64_t batch = batch_base + sub * kPack + index;
          if (batch < batch_size) grad[index] += projected[static_cast<int64_t>(pair) * batch_size + batch] * weight;
        }
      }
    }
  }
#pragma unroll
  for (int mask = kLanesPerEdge; mask < 32; mask <<= 1) {
#pragma unroll
    for (int index = 0; index < kPack; ++index) grad[index] += __shfl_xor_sync(0xffffffff, grad[index], mask);
  }
  if (lane < kLanesPerEdge) {
#pragma unroll
    for (int index = 0; index < kPack; ++index) partials[warp][sub * kPack + index] = grad[index];
  }
  __syncthreads();
  if (warp == 0 && batch_base + lane < batch_size) {
    float total = 0.0f;
#pragma unroll
    for (int source = 0; source < kWarps; ++source) total += partials[source][lane];
    spike_grad[(batch_base + lane) * n_pre + pre] =
        FromFloat<T>(total * AsFloat(*dampening));
  }
}
