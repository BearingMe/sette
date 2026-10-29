// M05 B4.4 / E5: native `rrlite_enc0_4x4_mvhi_ldr_folded` (Preset M, frame 1,
// reset, LDR).
//
// Source: the kernel's own PTX in `nvngx_dlss.dll` 310.7.0 (module
// `dlss-0154-00.ptx`), not d4r's reimplementation.  EXP-015 records the
// instruction behind every line below; this file implements the canonical
// reset frame, and the paths that frame does not exercise are listed there.
//
//   weights   param+0.  12416 B = 4096 (dither table) + 8192 (matrix) + 128 (bias).
//             The matrix is `ld.weak.global.ca.v4.u32` at
//             weights + lane*16 + 4096 + 512*kb, feeding the mma B operand:
//                 byte(k, n) = 4096 + 512*kb + 16*lane + 8*g + j
//                 kb = 4*(n/16) + k/32, lane = 4*(n%8) + (k%16)/4,
//                 g = (n%16)/8,        j = k%4 + 4*((k%32)/16)
//             which is `m_weights.layout_offsets('enc0')`, checked element for
//             element.  The bias is one f16 per column at 12288.
//   colour    param+80, point-sampled with `tex.base.2d` after the mirror fold.
//   tokens    param+48 = alloc1 + 0xFF0000, written with `st.global` at
//                 tokens + (c/32)*4177920 + ty*15360 + tx*32 + (c%32)
//             - 272 rows per plane, from the *padded* height at param+16 (544).
//
// A field token is one 2x2 pixel block; its four pixels are the embedding's four
// k32 steps, in the order (0,0), (0,1), (1,0), (1,1).  Each pixel contributes 32
// feature slots, and on a reset frame exactly 15 of them are nonzero: the texel's
// channels cycle r,g,b in ascending slot order.  The other 17 are the history,
// previous-feature and exposure slots, which `reset` zeroes.
//
// Accumulation: the bias seeds the f16 accumulator, then one f16 rounding per 32
// products - the mma chain.  The 32-term dot is summed exactly (double) and
// rounded once, which is what the CPU oracle does and what reproduced the
// captured field byte for byte.  A product of two e4m3 values is exact in f32;
// their sum is not (34 significant bits are reachable), so the double is not a
// refinement - it is the only way to round the f16 accumulator once.
//
// Epilogue: `max.f16x2`/`min.f16x2` to +-2*pi, `cvt.rn.satfinite.e4m3x2`, and the
// channel permutation the eight final accumulator chains impose - channel c is
// not column c.  The permutation is a bijection of 0..63 and is exercised by
// every byte of the output.

cbuffer Params : register(b0)
{
    int2 visible;       // param+56, the mirror bound (960, 540)
    int2 padded;        // param+16, sizes the token field (960, 544)
    uint frame;         // param+312, the dither table's frame index
    uint reset;         // param+316; this shader implements the reset path only
};

Buffer<uint> Weights : register(t0);        // 12416 B
Texture2D<float4> Colour : register(t1);    // 960x540, R16G16B16A16_FLOAT
RWBuffer<uint> Tokens : register(u0);       // 8 355 840 B, 2 planes

static const uint kTokensPerRow = 480;
static const uint kTokenRowWords = 3840;    // 15360 B
static const uint kPlaneWords = 1044480;    // 4 177 920 B

//: The 32 slots of one pixel and the colour channel each carries on a reset
//: frame (3 = exactly zero).  From the two `st.shared.v4.u32` stores and the
//: dither registers that feed them.
static const uint kSlotChannel[32] = {
    0u, 1u, 3u, 3u, 2u, 0u, 3u, 3u, 1u, 2u, 3u, 3u, 0u, 1u, 3u, 3u,
    2u, 0u, 3u, 3u, 1u, 2u, 3u, 3u, 0u, 1u, 3u, 3u, 2u, 3u, 3u, 3u,
};

//: Field channel -> mma column, from the epilogue's cvt/pack/store chain.
static const uint kColumn[64] = {
     0u,  1u, 16u, 17u,  2u,  3u, 18u, 19u,  4u,  5u, 20u, 21u,  6u,  7u, 22u, 23u,
     8u,  9u, 24u, 25u, 10u, 11u, 26u, 27u, 12u, 13u, 28u, 29u, 14u, 15u, 30u, 31u,
    32u, 33u, 48u, 49u, 34u, 35u, 50u, 51u, 36u, 37u, 52u, 53u, 38u, 39u, 54u, 55u,
    40u, 41u, 56u, 57u, 42u, 43u, 58u, 59u, 44u, 45u, 60u, 61u, 46u, 47u, 62u, 63u,
};

// ---------------------------------------------------------------------------
// binary16 helpers, shared with `downsample_cs.hlsl` and `dec0_cs.hlsl`.
// `RoundPair` is exact round-to-nearest-even of an f32 pair to f16.

float Pow2(int e)
{
    return asfloat(uint(e + 127) << 23);
}

float RoundPair(float high, float low)
{
    uint u = asuint(high);
    uint sign = (u >> 16) & 0x8000u;
    uint magnitude = u & 0x7FFFFFFFu;
    if (magnitude == 0u || magnitude >= 0x7F800000u)
        return high;
    float value = sign ? -high : high;
    float residual = sign ? -low : low;
    int exponent = int(magnitude >> 23) - 127;
    int quantum = max(exponent - 10, -24);
    float scaled = value * Pow2(-quantum);
    float whole = floor(scaled);
    float rest = scaled - whole;
    int n = int(whole);
    if (rest > 0.5f || (rest == 0.5f && (residual > 0.0f
                                         || (residual == 0.0f && (n & 1) != 0))))
        n += 1;
    if (exponent <= -15)
        return f16tof32(sign | uint(min(n, 1024)));
    if (n == 2048)
    {
        n = 1024;
        exponent += 1;
    }
    if (exponent > 15)
        return f16tof32(sign | 0x7C00u);
    return f16tof32(sign | (uint(exponent + 15) << 10) | uint(n - 1024));
}

float Half(float value) { return RoundPair(value, 0.0f); }

// Round an exact double to f16 once: the high f32 carries the value and the
// residual breaks an exact tie at the f16 midpoint.
float Half64(double value)
{
    precise float high = float(value);
    precise float low = float(value - double(high));
    return RoundPair(high, low);
}

// ---------------------------------------------------------------------------
// E4M3 (OCP): 1-4-3, bias 7, 0x7E = 448 is the largest finite magnitude.

float E4M3Value(uint code)
{
    uint s = code >> 7;
    uint e = (code >> 3) & 0xFu;
    uint m = code & 0x7u;
    if (e == 0u)
    {
        if (m == 0u)
            return s ? -0.0f : 0.0f;
        float v = float(m) * 0.001953125f;              // m * 2^-9
        return s ? -v : v;
    }
    if (e == 15u && m == 7u)
        return asfloat((s << 31) | 0x7FC00000u);        // NaN
    return asfloat((s << 31) | ((e + 120u) << 23) | (m << 20));
}

uint E4M3Code(float value)
{
    uint u = f32tof16(value);
    uint sign = (u >> 15) & 1u;
    uint magnitude = u & 0x7FFFu;
    if (magnitude == 0u)
        return sign << 7;
    uint biased = (magnitude >> 10) & 0x1Fu;
    uint mant = magnitude & 0x3FFu;
    if (biased == 0x1Fu)
        return (sign << 7) | 0x7Eu;                     // satfinite
    if (biased < 9u)                                    // below 2^-6
    {
        if (biased == 0u)
            return sign << 7;
        uint shift = 16u - biased;
        uint m = (1024u + mant) >> shift;
        uint rem = (1024u + mant) & ((1u << shift) - 1u);
        uint halfbit = 1u << (shift - 1u);
        if (rem > halfbit || (rem == halfbit && (m & 1u) != 0u))
            m += 1u;
        return (sign << 7) | min(m, 126u);
    }
    uint e4 = biased - 8u;
    uint m4 = mant >> 7;
    uint rem = mant & 0x7Fu;
    if (rem > 64u || (rem == 64u && (m4 & 1u) != 0u))
    {
        m4 += 1u;
        if (m4 == 8u)
        {
            m4 = 0u;
            e4 += 1u;
        }
    }
    if (e4 > 15u || (e4 == 15u && m4 == 7u))
        return (sign << 7) | 0x7Eu;
    return (sign << 7) | (e4 << 3) | m4;
}

// ---------------------------------------------------------------------------
// the blob

uint BlobByte(uint offset)
{
    return (Weights[offset >> 2] >> ((offset & 3u) * 8u)) & 0xFFu;
}

uint WeightByte(uint k, uint n)
{
    uint kb = 4u * (n >> 4) + (k >> 5);
    uint lane = 4u * (n & 7u) + ((k & 15u) >> 2);
    uint g = (n & 15u) >> 3;
    uint j = (k & 3u) + 4u * ((k & 31u) >> 4);
    return BlobByte(4096u + 512u * kb + 16u * lane + 8u * g + j);
}

float BiasValue(uint n)
{
    uint offset = 12288u + 2u * n;
    return f16tof32((Weights[offset >> 2] >> ((offset & 3u) * 8u)) & 0xFFFFu);
}

// `ld.global.u16` at weights + 2*((frame<<8) | (y&15)<<4 | (x&15)), `>> 9`.
uint DitherBits(uint2 pixel)
{
    uint index = (frame << 8) | ((pixel.y & 15u) << 4) | (pixel.x & 15u);
    uint offset = 2u * index;
    return ((Weights[offset >> 2] >> ((offset & 3u) * 8u)) & 0xFFFFu) >> 9u;
}

// The dither itself: `and 0xFC00FC00` / `or n` / `add.f16x2 h,(e|n)` /
// `sub.f16x2 -e` / `and 0xFF80FF80`.  The add is against the clamped value, not
// against its exponent - the distinction that the earlier model got wrong.
float Dither(float value, uint n)
{
    precise float v = isnan(value) ? 0.0f : min(value, 1024.0f);
    uint h = f32tof16(v);
    uint e = h & 0xFC00u;
    float added = RoundPair(f16tof32(h) + f16tof32(e | n), 0.0f);
    float t = RoundPair(added - f16tof32(e), 0.0f);
    return f16tof32(f32tof16(t) & 0xFF80u);
}

// ---------------------------------------------------------------------------
// the feature vector: 4 pixels x 32 slots = 128 k channels

float4 Texel(uint2 pixel)
{
    int2 p = int2(pixel);
    int2 s = int2(p.x < visible.x ? p.x : 2 * visible.x - p.x - 2,
                  p.y < visible.y ? p.y : 2 * visible.y - p.y - 2);
    return Colour.Load(int3(s, 0));
}

float Column(uint n, float fv[128])
{
    float acc = BiasValue(n);
    [unroll]
    for (uint step = 0u; step < 4u; ++step)
    {
        double sum = double(acc);
        [unroll]
        for (uint i = 0u; i < 32u; ++i)
        {
            uint k = step * 32u + i;
            sum += double(fv[k]) * double(E4M3Value(WeightByte(k, n)));
        }
        acc = Half64(sum);
    }
    return acc;
}

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint tokensPerRow = uint(padded.x) / 2u;
    uint rows = uint(padded.y) / 2u;
    uint token = id.x;
    uint ty = token / tokensPerRow;
    uint tx = token - ty * tokensPerRow;
    if (ty >= rows)
        return;

    // the four pixels of the token, in k-step order
    float fv[128];
    [unroll]
    for (uint sub = 0u; sub < 4u; ++sub)
    {
        uint2 pixel = uint2(2u * tx + (sub & 1u), 2u * ty + (sub >> 1u));
        float4 texel = Texel(pixel);
        uint n = DitherBits(pixel);
        [unroll]
        for (uint slot = 0u; slot < 32u; ++slot)
        {
            uint channel = kSlotChannel[slot];
            fv[sub * 32u + slot] = channel == 3u
                ? 0.0f
                : E4M3Value(E4M3Code(Dither(texel[channel], n)));
        }
    }

    uint codes[64];
    [unroll]
    for (uint c = 0u; c < 64u; ++c)
    {
        float acc = Column(kColumn[c], fv);
        acc = min(max(acc, -6.28125f), 6.28125f);       // max/min.f16x2, +-2*pi
        codes[c] = E4M3Code(acc);
    }

    uint base = ty * kTokenRowWords + tx * 8u;
    [unroll]
    for (uint plane = 0u; plane < 2u; ++plane)
    {
        uint at = base + plane * kPlaneWords;
        [unroll]
        for (uint w = 0u; w < 8u; ++w)
        {
            uint c = plane * 32u + w * 4u;
            Tokens[at + w] = codes[c] | (codes[c + 1u] << 8)
                           | (codes[c + 2u] << 16) | (codes[c + 3u] << 24);
        }
    }
}
