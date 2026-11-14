// Copyright © 2026 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sort.h"

using namespace metal;

template <typename T, short Threads, short Items>
[[kernel, max_total_threads_per_threadgroup(Threads)]] void
argpartition_candidates(
    const device T* in [[buffer(0)]],
    device T* candidates [[buffer(1)]],
    device uint* indices [[buffer(2)]],
    device uint* out [[buffer(3)]],
    const constant uint& width [[buffer(4)]],
    const constant uint& candidate_width [[buffer(5)]],
    const constant uint& edge_size [[buffer(6)]],
    const constant uint& upper [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  constexpr uint tile_size = Threads * Items;
  using TileSort = BlockMergeSort<T, uint, true, Threads, Items, LessThan<T>>;
  threadgroup T values[tile_size];
  threadgroup uint positions[tile_size];

  uint start = group.x * tile_size;
  uint count = min(tile_size, width - start);
  uint row = group.y * width;
  for (uint i = lid.x; i < tile_size; i += Threads) {
    uint index = start + i;
    values[i] = i < count ? in[row + index] : LessThan<T>::init;
    positions[i] = index;
    if (i < count) {
      out[row + index] = index;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  TileSort::sort(values, positions, count, lid);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint edge_count = min(edge_size, count);
  if (lid.x < edge_count) {
    uint slot = group.y * candidate_width + group.x * edge_size + lid.x;
    uint position = (upper ? count - edge_count : 0) + lid.x;
    candidates[slot] = values[position];
    indices[slot] = positions[position];
  }
}

[[kernel]] void argpartition_permute(
    const device uint* indices [[buffer(0)]],
    const device uint* order [[buffer(1)]],
    device uint* out [[buffer(2)]],
    const constant uint& width [[buffer(3)]],
    const constant uint& candidate_width [[buffer(4)]],
    const constant uint& edge_size [[buffer(5)]],
    const constant uint& upper [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
  threadgroup uint selected[64];
  uint edge_start = upper ? width - edge_size : 0;
  uint candidate_start = upper ? candidate_width - edge_size : 0;
  out += row * width;
  indices += row * candidate_width;
  order += row * candidate_width + candidate_start;

  if (lid < edge_size) {
    uint index = indices[order[lid]];
    selected[lid] = index;
    out[edge_start + lid] = index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Move displaced indices into the slots vacated by the selected indices.
  if (lid == 0) {
    ulong occupied = 0;
    for (uint i = 0; i < edge_size; ++i) {
      uint slot = selected[i] - edge_start;
      if (slot < edge_size) {
        occupied |= 1ul << slot;
      }
    }
    uint next_slot = 0;
    for (uint i = 0; i < edge_size; ++i) {
      uint index = selected[i];
      if (index - edge_start >= edge_size) {
        while ((occupied >> next_slot) & 1ul) {
          ++next_slot;
        }
        out[index] = edge_start + next_slot++;
      }
    }
  }
}

instantiate_kernel(
    "argpartition_candidates_float32", argpartition_candidates, float, 512, 4)
instantiate_kernel(
    "argpartition_candidates_float16", argpartition_candidates, half, 512, 4)
instantiate_kernel(
    "argpartition_candidates_bfloat16",
    argpartition_candidates,
    bfloat16_t,
    512,
    4)
