#include <metal_stdlib>
using namespace metal;

// Each SIMD group computes one (outer, head, query) row
// and each of its 32 lanes owns one value channel.
kernel void triangle_attention_d32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* mask_bias [[buffer(3)]],
    device const float* triangle_bias [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& outer_size [[buffer(6)]],
    constant uint& head_count [[buffer(7)]],
    constant uint& query_length [[buffer(8)]],
    constant uint& key_length [[buffer(9)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]])
{
    constexpr uint D = 32;
    constexpr uint ROWS_PER_GROUP = 4;
    const uint row = threadgroup_position.x * ROWS_PER_GROUP + simdgroup_index;
    const uint row_count = outer_size * head_count * query_length;
    if (row >= row_count) return;

    const uint query_index = row % query_length;
    const uint outer_head = row / query_length;
    const uint head = outer_head % head_count;
    const uint outer = outer_head / head_count;

    // [outer, head, token, D]
    const uint q_offset =
        ((outer * query_length + query_index) * head_count + head) * D + lane;

    float running_max = -INFINITY;
    float denominator = 0.0f;
    float accumulator = 0.0f;

    for (uint key_index = 0; key_index < key_length; ++key_index) {
        const uint kv_offset =
            ((outer * key_length + key_index) * head_count + head) * D + lane;
        const float partial = q[q_offset] * k[kv_offset];
        const float dot = simd_sum(partial);

        const float score = dot
            + mask_bias[outer * key_length + key_index]
            + triangle_bias[(head * query_length + query_index) * key_length
                            + key_index];
        const float next_max = max(running_max, score);
        const float old_scale = exp(running_max - next_max);
        const float new_scale = exp(score - next_max);
        accumulator = accumulator * old_scale + new_scale * v[kv_offset];
        denominator = denominator * old_scale + new_scale;
        running_max = next_max;
    }

    const uint out_offset =
        ((outer * query_length + query_index) * head_count + head) * D + lane;
    out[out_offset] = accumulator / denominator;
}
