// M15-B: fuse the post-MLP GELU materialization and the symmetric per-token
// INT8 encoder into one dispatch.
//
// The arithmetic is the accepted pair, step for step:
//
//   z  = HalfAt(MLP-up f16)
//   g  = GeluQ(z)                       (swin_mlp_gelu_cs.hlsl)
//   mx = max(abs(g))                    (swin_q8_quant_cs.hlsl, four lanes)
//   scale = mx * (1/127)
//   code = round(g * (127/mx))          saturated to [-127, 127]
//
// The row maximum still uses the accepted four-lane groupshared reduction, so
// no token is split across threadgroups.  GELU is evaluated once: each lane
// writes its own chunk of G while accumulating the row maximum, then reads the
// same chunk back for the code pass.  The store is required for the diagnostic
// tensor anyway, and a thread only reads values it wrote itself, so no extra
// barrier or cross-threadgroup synchronization is needed.
//
// G is therefore byte-identical to the separate GELU dispatch, and the code
// pass sees exactly the f16 values the accepted quantizer reads.

#include "swin_q8_common.hlsli"

cbuffer Q8Params : register(b0)
{
    uint Tokens;
    uint K;
    uint N;
    uint Unused;
};

Buffer<uint> A : register(t0);       // MLP-up f16 output, [tokens][K]
RWBuffer<uint> G : register(u0);     // GELU/E4M3 f16, same layout (diagnostic)
RWBuffer<uint> Codes : register(u1); // int8, four per word
RWBuffer<uint> Scales : register(u2); // f32 per token

#define Q8_LANES 4

groupshared float g_rowmax[64 * Q8_LANES];

// ---------------------------------------------------------------- f16 helpers
// Exact copies of swin_layer_cs.hlsl.  The accepted GELU is an f16 operation
// chain with one rounding per operation; reimplementing it in f32 would move
// values across the E4M3 boundaries and change the INT8 codes.

float R16(float v) { return f16tof32(ToHalfBits(v)); }

float R16D(double v)
{
    float f = (float)v;
    uint u = asuint(f);
    if ((u & 0x1FFFu) == 0x1000u)
    {
        double d = v - (double)f;
        if (d != 0.0)
        {
            bool up = ((d > 0.0) == ((u & 0x80000000u) == 0u));
            return R16(asfloat(up ? u + 1u : u - 1u));
        }
    }
    return R16(f);
}

float Add16(float a, float b) { return R16D((double)a + (double)b); }
float Mul16(float a, float b) { return R16D((double)a * (double)b); }

// q8_exact from swin_common.h: e4m3 RNE satfinite, NaN payload kept
uint Q8Bits(uint u)
{
    uint m = u & 0x7FFFu;
    uint r = (m + 0x3Fu + ((m >> 7) & 1u)) & 0x7F80u;
    r = r < 0x5F00u ? r : 0x5F00u;
    float mag = f16tof32(m);
    float subf = R16(Add16(mag, 2.0f) - 2.0f);
    uint sub = ToHalfBits(subf);
    r = m < 0x2400u ? sub : r;
    r = m > 0x7C00u ? m : r;
    return (u & 0x8000u) | r;
}

float Q8(float v) { return f16tof32(Q8Bits(ToHalfBits(v))); }

float HalfAtRW(RWBuffer<uint> b, uint i)
{
    uint w = b[i >> 1];
    return f16tof32((i & 1u) ? (w >> 16) : (w & 0xFFFFu));
}

float GeluQ(float z)
{
    float cz = z < -2.0f ? -2.0f : (z > 2.0f ? 2.0f : z);
    float az = abs(cz);
    float inner = Add16(0.412109375f, -Mul16(0.0810546875f, az));
    float gl = Mul16(z, Add16(0.5f, Mul16(cz, inner)));
    return Q8(gl);
}

[numthreads(256, 1, 1)]
void GeluQuant(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    uint local = gi >> 2;
    uint lane = gi & (Q8_LANES - 1u);
    uint tok = (id.x >> 8) * 64u + local;
    bool active = tok < Tokens;
    uint safe = active ? tok : (Tokens > 0u ? Tokens - 1u : 0u);

    uint chunk = K >> 2;
    uint base = safe * K + lane * chunk;
    float mx = 0.0f;
    for (uint c = 0; c < chunk; c += 2u)
    {
        float v0 = GeluQ(HalfAt(A, base + c));
        float v1 = GeluQ(HalfAt(A, base + c + 1u));
        mx = max(mx, max(abs(v0), abs(v1)));
        if (active)
            G[(base + c) >> 1] = ToHalfBits(v0) | (ToHalfBits(v1) << 16);
    }

    g_rowmax[local * Q8_LANES + lane] = mx;
    GroupMemoryBarrierWithGroupSync();
    float m0 = g_rowmax[local * Q8_LANES + 0u];
    float m1 = g_rowmax[local * Q8_LANES + 1u];
    float m2 = g_rowmax[local * Q8_LANES + 2u];
    float m3 = g_rowmax[local * Q8_LANES + 3u];
    mx = max(max(m0, m1), max(m2, m3));
    if (!active)
        return;
    if (mx == 0.0f)
        mx = 1.0f;
    float invf = 127.0f / mx;
    double residual = 127.0 - (double)invf * (double)mx;
    float inv = (float)((double)invf + residual / (double)mx);
    if (lane == 0u)
        Scales[tok] = asuint(mx * 0.0078740157f);

    uint wordBase = base >> 2;
    for (uint g = 0; g < (chunk >> 2); ++g)
    {
        uint i = base + 4u * g;
        float v0 = HalfAtRW(G, i + 0u);
        float v1 = HalfAtRW(G, i + 1u);
        float v2 = HalfAtRW(G, i + 2u);
        float v3 = HalfAtRW(G, i + 3u);
        Codes[wordBase + g] = (uint)(QuantCode(v0 * inv) & 0xFF)
                            | ((uint)(QuantCode(v1 * inv) & 0xFF) << 8)
                            | ((uint)(QuantCode(v2 * inv) & 0xFF) << 16)
                            | ((uint)(QuantCode(v3 * inv) & 0xFF) << 24);
    }
}
