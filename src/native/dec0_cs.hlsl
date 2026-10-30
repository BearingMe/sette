// M05 B4.3 / D7: native `rrlite_dec0_4x4_folded` (Preset M, render 960x544).
//
// Source: the kernel's own PTX in `nvngx_dlss.dll` 310.7.0 (module
// `dlss-0162-00.ptx`), not d4r's reimplementation.  The PTX fixes the operand
// mapping by address arithmetic (EXP-014):
//
//   weights   param+0.  `ld.weak.global.ca.v4.u32` at weights + lane*16 + 512*kb
//             feeds `mma.m16n8k32.row.col.f16.e4m3.e4m3.f16` as the B operand
//             (8 bytes per lane at lane*16 + 8*g).  For B, k = 4*tig + j%4 +
//             16*(j/4), n = groupID, so the byte of element (k, n) is
//                 512*kb + 16*lane + 8*g + j
//                 kb = 2*(n/16) + k/32, lane = 4*(n%8) + (k%16)/4,
//                 g = (n%16)/8,        j = k%4 + 4*((k%32)/16)
//             which is exactly `m_weights.layout_offsets('dec0')` - checked, not
//             assumed.
//   bias      weights+8192, f16 index == output column (128 columns).
//   tokens    param+24, `ld.global.u32` at
//                 tokens + (c/32)*4177920 + ty*15360 + tx*32 + (c%32)
//             i.e. 32 bytes per token per channel plane, 480 tokens per row,
//             272 rows per plane (the parameter block's height is 544, the
//             padded render height - reading it as 540 was EXP-013's error).
//
// Each of the four 2x2 sub-positions uses 20 of the 128 columns; column 0 is
// the accumulator that produced the first gate, which is what ties the geometry
// together (validated at 100.000% byte agreement, EXP-014).
//
// Accumulation: bias is the initial f16 accumulator, then one f16 rounding per
// 32 products, which is d4r's published `mma_chain` contract and the arithmetic
// EXP-010 validated.  A product of two e4m3 values is exact in f32; the 32-term
// sum is done in f64 so no intermediate rounding is introduced.
//
// Epilogue arithmetic is transcribed instruction by instruction, including the
// places where the kernel is *less* accurate than a textbook formula: the
// sigmoid rounds to f16 before and after, the softmax normalises with a
// reciprocal, and the RGBA16F stores truncate toward zero (EXP-014 measured
// that: round-to-nearest reproduces 47% of the covariance values, truncation
// 99.98%).

cbuffer Params : register(b0)
{
    int2 size;          // render size, padded (960, 544)
    int2 pad0;
    float sigma;        // 8.0
    float rho;          // 0.0786
    int reset;          // 1 on the first frame
    int pad1;
};

Buffer<uint> Weights : register(t0);     // 8448 B
Buffer<uint> Tokens : register(t1);      // 8 355 840 B
RWTexture2D<uint> GateOut : register(u0);
RWTexture2D<uint> MixOut : register(u1);
RWTexture2D<uint> FeatOut : register(u2);
RWTexture2D<float4> CovOut : register(u3);

static const uint kTokensPerRow = 480;
static const uint kTokenRowBytes = 15360;
static const uint kPlaneBytes = 4177920;

// ---------------------------------------------------------------------------
// binary16 helpers.  `RoundPair` is the one `downsample_cs.hlsl` validated byte
// for byte: exact round-to-nearest-even of an f32 pair to f16, in integers.

float Pow2(int e)
{
    return asfloat(uint(e + 127) << 23);
}

float2 TwoSum(float a, float b)
{
    precise float sum = a + b;
    precise float bb = sum - a;
    precise float ea = a - (sum - bb);
    precise float eb = b - bb;
    return float2(sum, ea + eb);
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

// An f16 value written to an RGBA16F surface: the store truncates toward zero.
float HalfTrunc(float value)
{
    uint u = asuint(value);
    uint sign = u & 0x80000000u;
    uint magnitude = u & 0x7FFFFFFFu;
    if (magnitude == 0u || magnitude >= 0x7F800000u)
        return value;
    int exponent = int(magnitude >> 23) - 127;
    if (exponent > 15)
        return asfloat(sign | 0x7F800000u);
    if (exponent < -14)
    {
        // f16 subnormal: the quantum is 2^-24, and truncation is now a
        // truncating scale rather than a mantissa mask
        uint n = uint(asfloat(magnitude) * Pow2(24));
        return f16tof32((sign >> 16) | min(n, 1023u));
    }
    return f16tof32((sign >> 16) | (uint(exponent + 15) << 10)
                    | ((magnitude >> 13) & 0x3FFu));
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
        // subnormal: 2^(1-7) * (0 + m/8) = m * 2^-9.  There is no implicit
        // leading 1, so the value is built by an exact scale, not a mantissa
        // mask - 17 of the 64 weights of column 0 are subnormal codes.
        if (m == 0u)
            return s ? -0.0f : 0.0f;
        float v = float(m) * 0.001953125f;              // m * 2^-9
        return s ? -v : v;
    }
    if (e == 15u && m == 7u)
        return asfloat((s << 31) | 0x7FC00000u);     // NaN
    return asfloat((s << 31) | ((e + 120u) << 23) | (m << 20));
}

// `cvt.rn.satfinite.e4m3x2.f16x2`: nearest e4m3 of an f16 value, ties to even,
// saturating at 448.  The argument is always an exact binary16 value (it comes
// from `Half`), so `f32tof16` cannot round it differently.
uint E4M3Code(float value)
{
    uint u = f32tof16(value);
    uint sign = (u >> 15) & 1u;
    uint magnitude = u & 0x7FFFu;
    if (magnitude == 0u)
        return sign << 7;
    uint biased = (magnitude >> 10) & 0x1Fu;
    uint mant = magnitude & 0x3FFu;
    if (biased == 0x1Fu)                              // inf / NaN
        return (sign << 7) | 0x7Eu;                    // satfinite
    if (biased < 9u)                                   // below 2^-6: e4m3 subnormal
    {
        // value = (1024 + mant) * 2^(biased-25); the e4m3 quantum is 2^-9, so
        // the quotient is (1024 + mant) >> (16 - biased)
        if (biased == 0u)                              // f16 subnormal: far below 2^-9
            return sign << 7;
        uint shift = 16u - biased;
        uint m = (1024u + mant) >> shift;
        uint rem = (1024u + mant) & ((1u << shift) - 1u);
        uint halfbit = 1u << (shift - 1u);
        if (rem > halfbit || (rem == halfbit && (m & 1u) != 0u))
            m += 1u;
        return (sign << 7) | min(m, 126u);
    }
    uint e4 = biased - 8u;                             // e4m3 exponent field
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
        return (sign << 7) | 0x7Eu;                     // saturate
    return (sign << 7) | (e4 << 3) | m4;
}

// ---------------------------------------------------------------------------
// the matrix

uint WeightByte(uint k, uint n)
{
    uint kb = 2u * (n >> 4) + (k >> 5);
    uint lane = 4u * (n & 7u) + ((k & 15u) >> 2);
    uint g = (n & 15u) >> 3;
    uint j = (k & 3u) + 4u * ((k & 31u) >> 4);
    uint offset = 512u * kb + 16u * lane + 8u * g + j;
    return (Weights[offset >> 2] >> ((offset & 3u) * 8u)) & 0xFFu;
}

float TokenValue(uint ty, uint tx, uint c)
{
    uint offset = (c >> 5) * kPlaneBytes + ty * kTokenRowBytes + tx * 32u + (c & 31u);
    return E4M3Value((Tokens[offset >> 2] >> ((offset & 3u) * 8u)) & 0xFFu);
}

float BiasValue(uint n)
{
    uint offset = 8192u + 2u * n;
    return f16tof32((uint)((Weights[offset >> 2] >> ((offset & 3u) * 8u)) & 0xFFFFu));
}

// One column, with the published accumulator contract: the bias seeds the f16
// accumulator, then a f16 rounding after each 32 products.
float Column(uint ty, uint tx, uint n)
{
    float acc = BiasValue(n);
    [unroll]
    for (uint step = 0u; step < 2u; ++step)
    {
        double sum = double(acc);
        [unroll]
        for (uint i = 0u; i < 32u; ++i)
        {
            uint k = step * 32u + i;
            sum += double(TokenValue(ty, tx, k)) * double(E4M3Value(WeightByte(k, n)));
        }
        acc = Half(float(sum));
    }
    return acc;
}

// ---------------------------------------------------------------------------
// epilogue

// EXP-025 supplies only the expansion value; the default still calls Column.
#ifndef DEC0_LOAD_COLUMN
#define DEC0_LOAD_COLUMN Column
#endif

float Sigmoid(float v)
{
    float c = Half(clamp(v, -20.0f, 20.0f));
    float e = exp2(-1.4426951f * c);
    return Half(1.0f / (1.0f + e));
}

uint Unorm8(float v)
{
    return uint(floor(clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f));
}

uint Snorm8(float v)
{
    float x = clamp(v, -1.0f, 1.0f);
    uint code = uint(floor(abs(x) * 127.0f + 0.5f));
    return (x < 0.0f) ? (256u - code) & 0xFFu : code;
}

float Correlation(float t)
{
    precise float s = t * t;
    float mag = abs(t);
    float big = 1.0f / (exp2(mag * 2.88539f) + 1.0f);
    big = big * -2.0f + 1.0f;
    big = (mag >= 9.0109138f) ? 1.0f : big;
    float poly = 0.01573968306183815f * s + -0.05230396240949631f;
    poly = poly * s + 0.13315297663211823f;
    poly = poly * s + -0.33332768082618713f;
    poly = poly * s;
    poly = poly * t + t;
    return (mag >= 0.6f) ? ((t < 0.0f) ? -big : big) : poly;
}

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint token = id.x;
    uint ty = token / kTokensPerRow;
    uint tx = token % kTokensPerRow;
    if (ty >= uint(size.y) / 2u || tx >= uint(size.x) / 2u)
        return;

    [unroll]
    for (uint sub = 0u; sub < 4u; ++sub)
    {
        uint base = sub * 32u;
        float v[20];
        [unroll]
        for (uint i = 0u; i < 20u; ++i)
            v[i] = Half(E4M3Value(E4M3Code(DEC0_LOAD_COLUMN(ty, tx, base + i))));

        int2 pix = int2(2 * int(tx) + int(sub & 1u), 2 * int(ty) + int(sub >> 1));

        uint g0 = Unorm8(Sigmoid(v[0]));
        uint g1 = Unorm8(Sigmoid(v[1]));
        uint g2 = Unorm8(Sigmoid(v[10]));
        GateOut[pix] = g0 | (g1 << 8) | (g2 << 16);

        float mx = max(max(v[11], v[12]), v[13]);
        float e0 = exp2((v[11] - mx) * 1.4426951f);
        float e1 = exp2((v[12] - mx) * 1.4426951f);
        float e2 = exp2((v[13] - mx) * 1.4426951f);
        float total = e2 + (e0 + e1);
        float rcp = 1.0f / total;
        uint m0 = Unorm8(e0 * rcp);
        uint m1 = Unorm8(e1 * rcp);
        uint m2 = Unorm8(e2 * rcp);
        MixOut[pix] = m0 | (m1 << 8) | (m2 << 16);

        // recurrent feature: (1-g)*history + g*tanh(v); frame 1 is a reset
        // frame, so the history term is exactly zero
        uint f0, f1, f2, f3;
        {
            float g = Half(Half(0.5f * Half(tanh(Half(0.5f * v[6])))) + 0.5f);
            float t = Half(tanh(v[2]));
            f0 = Snorm8(Half(g * t));
            g = Half(Half(0.5f * Half(tanh(Half(0.5f * v[7])))) + 0.5f);
            t = Half(tanh(v[3]));
            f1 = Snorm8(Half(g * t));
            g = Half(Half(0.5f * Half(tanh(Half(0.5f * v[8])))) + 0.5f);
            t = Half(tanh(v[4]));
            f2 = Snorm8(Half(g * t));
            g = Half(Half(0.5f * Half(tanh(Half(0.5f * v[9])))) + 0.5f);
            t = Half(tanh(v[5]));
            f3 = Snorm8(Half(g * t));
        }
        FeatOut[pix] = f0 | (f1 << 8) | (f2 << 16) | (f3 << 24);

        float a = sigma * Sigmoid(v[14]);
        float b = sigma * Sigmoid(v[15]);
        float t = v[16] * rho;
        float aa = a * a;
        float bb = b * b;
        float ab = (a * b) * Correlation(t);
        CovOut[pix] = float4(HalfTrunc(aa), HalfTrunc(bb), HalfTrunc(ab), 0.0f);
    }
}
