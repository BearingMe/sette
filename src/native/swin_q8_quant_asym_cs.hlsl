// M13: asymmetric per-token INT8 encoding with an explicit offset.
//
//   in   A       f16 pairs, row-major [tokens][K]
//   out  Codes   int8, four per word,  dequant = code * scale + offset
//        Scales  f32 per token, (max - min) / 255
//        Offsets f32 per token, min + 128 * scale
//
// Codes are signed int8 in [-128, 127], so the existing signed `dot4` reduction
// is unchanged.  The offset moves the reconstruction range to the row's true
// [min, max], which matters where the operand is not symmetric about zero:
// post-GELU activations are mostly non-negative and softmax probabilities are
// entirely non-negative.  Applying the offset costs one per-output correction
// term in the consumer, not an extra MAC.
//
// The 4-lane-per-row split is the accepted EXP-040 structure: `min` and `max`
// are exact and associative, and each element's code depends only on its own
// value and the row scale, so the split is bit-identical to one lane per row.
// The scale/inverse arithmetic mirrors the accepted quantizer: the inverse is
// made exact to f64 and rounded once to f32, so a 1-ulp reciprocal cannot move
// a code across a rounding boundary.

#include "swin_q8_common.hlsli"

cbuffer Q8Params : register(b0)
{
    uint Tokens;
    uint K;
    uint N;
    uint Unused;
};

Buffer<uint> A : register(t0);
RWBuffer<uint> Codes : register(u0);
RWBuffer<uint> Scales : register(u1);
RWBuffer<uint> Offsets : register(u2);

#define Q8_LANES 4

groupshared float g_rowmin[64 * Q8_LANES];
groupshared float g_rowmax[64 * Q8_LANES];

int AsymCode(float r)
{
    return clamp((int)round(r) - 128, -128, 127);
}

[numthreads(256, 1, 1)]
void Quantize(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    uint local = gi >> 2;
    uint lane = gi & (Q8_LANES - 1u);
    uint tok = (id.x >> 8) * 64u + local;
    bool active = tok < Tokens;
    uint safe = active ? tok : (Tokens > 0u ? Tokens - 1u : 0u);

    uint chunk = K >> 2;
    uint base = safe * K + lane * chunk;
    float lo = 1.0e30f;
    float hi = -1.0e30f;
    for (uint c = 0; c < chunk; ++c)
    {
        float v = HalfAt(A, base + c);
        lo = min(lo, v);
        hi = max(hi, v);
    }

    uint slot = local * Q8_LANES + lane;
    g_rowmin[slot] = lo;
    g_rowmax[slot] = hi;
    GroupMemoryBarrierWithGroupSync();
    lo = min(min(g_rowmin[local * Q8_LANES + 0u], g_rowmin[local * Q8_LANES + 1u]),
             min(g_rowmin[local * Q8_LANES + 2u], g_rowmin[local * Q8_LANES + 3u]));
    hi = max(max(g_rowmax[local * Q8_LANES + 0u], g_rowmax[local * Q8_LANES + 1u]),
             max(g_rowmax[local * Q8_LANES + 2u], g_rowmax[local * Q8_LANES + 3u]));
    if (!active)
        return;

    float range = hi - lo;
    if (!(range > 0.0f))
        range = 1.0f;
    float scale = range * (1.0f / 255.0f);
    // 1/scale, exact to f64 then rounded once to f32 (same correction as the
    // accepted symmetric quantizer).
    float invf = 1.0f / scale;
    double residual = 1.0 - (double)invf * (double)scale;
    float inv = (float)((double)invf + residual / (double)scale);
    if (lane == 0u)
    {
        Scales[tok] = asuint(scale);
        Offsets[tok] = asuint(lo + 128.0f * scale);
    }

    uint wordBase = base >> 2;
    for (uint g = 0; g < (chunk >> 2); ++g)
    {
        uint word = 0;
        for (uint j = 0; j < 4u; ++j)
        {
            float v = HalfAt(A, base + 4u * g + j);
            word |= (uint)(AsymCode((v - lo) * inv) & 0xFF) << (8u * j);
        }
        Codes[wordBase + g] = word;
    }
}
