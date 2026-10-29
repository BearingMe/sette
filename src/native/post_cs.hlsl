// M05 B4.5: native `rrlite_post_3_1_mvhi_ldr_folded` (Preset M, reconstruction
// 2880x1620 from render 960x540, LDR).
//
// Source: the kernel's own PTX in `nvngx_dlss.dll` 310.7.0 (module
// `dlss-0145-00.ptx`), not d4r's reimplementation.  EXP-016 records the
// instruction behind each step; this file implements the captured frame
// (`reset = 1`, zero motion), and the paths that frame does not exercise are
// listed there.
//
//   tile      A 16x16 block owns 16 reconstruction columns and 32 rows.  Its
//             first 160 threads each fetch one render texel and write four f16
//             into each of three 640-half arrays.  Tile (col, row) is render
//             (r1 + col - 2, r2 + row - 2), clamped, with
//             r1 = trunc(scale_x*16*bx), r2 = trunc(scale_y*32*by).
//   gaussians The 3x3 taps of an output pixel sample `base + (i-1, j-1)`,
//             base = floor(pos - 0.5); the exponent is f16 throughout
//                 e = f16(f16(dx^2*xx) + f16(dy^2*yy)) + f16(f16(dx*dy)*xy)
//             and w = exp2(f32(e)); the nine accumulators are f16 and W is the
//             f32 sum of the unrounded weights in tap order.  The odd row of a
//             thread's pair keeps the even row's three texel rows and adjusts
//             its own dy by -1 when the two `floor(pos-0.5)` differ.
//   blend     g0 = sat(gain*W*gate.x); gates fma(g0, 1-gate.y/z, gate.y/z);
//             outputs fma(cur-hist, gate, hist).  `reset` (and an off-image
//             history) forces hist to the colour texel at floor(pos), which is
//             what this frame does.
//   outputs   outA -> the 2880x1620 RGBA16F surface, outB -> the packed
//             11-11-10 word buffer, and the 1440x810 surface gets the 10-bit
//             mean of outA over the row pair and the neighbouring column.
//
// The f16 surface stores truncate toward zero: `F16Trunc` reproduces the
// captured bytes (EXP-016 measures 99.998% agreement with truncation and 55.8%
// with round-to-nearest), so the surface is written through an R32_UINT view,
// two words per pixel, as dec0's byte surfaces are.
//
// The half-resolution surface is 10-bit: its captured values are exactly k/1023
// (EXP-016), so the mean is quantised before the store rather than written raw.

cbuffer Params : register(b0)
{
    int2 visible;        // param+148, the render size (960, 540)
    int2 recon;          // param+140, the reconstruction size (2880, 1620)
    int2 display;        // param+192, the motion texture's size (1920, 1080)
    int2 colourOrigin;   // param+8
    float scaleX;        // param+124
    float scaleY;        // param+128
    float offsetX;       // param+132
    float offsetY;       // param+136
    float gain;          // param+120
    float2 mvScale;      // param+200
    float2 mvOffset;     // param+208
    uint reset;          // param+168
};

Texture2D<float4> Colour : register(t0);     // 960x540 RGBA16F
Texture2D<float4> Gate : register(t1);       // 960x540 RGBA8 UNORM
Texture2D<float4> Mix : register(t2);        // 960x540 RGBA8 UNORM
Texture2D<float4> Cov : register(t3);        // 960x540 RGBA16F
Texture2D<float2> Motion : register(t4);     // 960x540 RG16F

RWBuffer<uint> Packed : register(u0);        // 2880*1620 words
RWTexture2D<uint> History : register(u1);    // 5760x1620 R32_UINT, 2 words/px
RWTexture2D<float4> HalfOut : register(u2);  // 1440x810 RGBA32F

static const float COV_XXYY = -0.7213249802589417f;    // 0fBF38A8C1
static const float COV_XY = -1.4426499605178833f;      // 0fBFB8A8C1
static const float SAT_MAX = 32752.0f;                 // 0f46FFE000
static const float RC_FLOOR = 1.000000013351432e-10f;  // 0f2EDBE6FF
static const float INV_255 = 0.003921568859368563f;    // 0f3B808081
static const float C_1_1 = 1.100000023841858f;         // 0f3F8CCCCD
static const float C_M0_05 = -0.05000000074505806f;    // 0fBD4CCCCD

// ---------------------------------------------------------------------------
// binary16 helpers, shared with `downsample_cs.hlsl` and `dec0_cs.hlsl`.

float Pow2(int e) { return asfloat(uint(e + 127) << 23); }

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

// One rounding of an exact double to f16.
float Half64(double value)
{
    precise float high = float(value);
    precise float low = float(value - double(high));
    return RoundPair(high, low);
}

// `fma.rn.f16` on f16 operands: the product is exact in double, so the sum is
// rounded once.
float F16Fma(float a, float b, float c)
{
    return Half64(double(a) * double(b) + double(c));
}

// `fma.rn.f32`: one rounding of a*b + c, independent of compiler contraction.
float Fma(float a, float b, float c)
{
    return float(double(a) * double(b) + double(c));
}

// f32 -> f16 bits, truncating toward zero.
uint F16Trunc(float v)
{
    uint bits = f32tof16(v);
    if (v != 0.0f && abs(f16tof32(bits)) > abs(v))
    {
        uint sign = bits & 0x8000u;
        uint magnitude = bits & 0x7FFFu;
        if (magnitude > 0u)
            bits = sign | (magnitude - 1u);
    }
    return bits;
}

uint Unorm10(float v)
{
    float scaled = floor(v * 1023.0f + 0.5f);
    return uint(clamp(scaled, 0.0f, 1023.0f));
}

// The captured surface holds the exact IEEE quotient k/1023.  dxc turns a
// float division by a constant into a reciprocal multiply even under `precise`,
// which lands one ulp low for 0.08% of codes, so the quotient is formed in
// double and rounded once to float.
float Unorm10Value(uint code)
{
    return float(double(code) / 1023.0);
}

// ---------------------------------------------------------------------------

groupshared float Tile0[640];    // colour.rgb, mix.r
groupshared float Tile1[640];    // mix.g, mix.b, gate.r, gate.g
groupshared float Tile2[640];    // gate.z, cov.x*k, cov.y*k, cov.z*2k
groupshared float ColumnSum[256 * 3];

int2 FoldColour(int2 rel)
{
    return int2(rel.x > 0 ? min(colourOrigin.x + rel.x, colourOrigin.x + visible.x - 1)
                          : colourOrigin.x,
                rel.y > 0 ? min(colourOrigin.y + rel.y, colourOrigin.y + visible.y - 1)
                          : colourOrigin.y);
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint lane = tid.x + tid.y * 16u;
    uint x = gid.x * 16u + tid.x;
    uint y = 2u * (gid.y * 16u + tid.y);
    int2 origin = int2(int(scaleX * float(gid.x * 16u)),
                       int(scaleY * float(2u * gid.y * 16u)));

    // --- tile fill: the first 160 threads, one render texel each -------------
    if (lane < 160u)
    {
        uint row = lane / 10u;
        uint col = lane - row * 10u;
        int2 rel = int2(origin.x + int(col) - 2, origin.y + int(row) - 2);
        int2 at = FoldColour(rel);
        float4 c = Colour.Load(int3(at, 0));
        float4 m = Mix.Load(int3(at, 0));
        float4 g = Gate.Load(int3(at, 0));
        float4 v = Cov.Load(int3(at, 0));

        float3 col3;
        col3.x = (isnan(c.x) || isinf(c.x)) ? 0.0f : min(c.x, SAT_MAX);
        col3.y = (isnan(c.y) || isinf(c.y)) ? 0.0f : min(c.y, SAT_MAX);
        col3.z = (isnan(c.z) || isinf(c.z)) ? 0.0f : min(c.z, SAT_MAX);

        uint e = lane * 4u;
        Tile0[e + 0u] = Half(col3.x);
        Tile0[e + 1u] = Half(col3.y);
        Tile0[e + 2u] = Half(col3.z);
        Tile0[e + 3u] = Half(m.x);
        Tile1[e + 0u] = Half(m.y);
        Tile1[e + 1u] = Half(m.z);
        Tile1[e + 2u] = Half(g.x);
        Tile1[e + 3u] = Half(g.y);
        Tile2[e + 0u] = Half(g.z);
        Tile2[e + 1u] = Half(v.x * COV_XXYY);
        Tile2[e + 2u] = Half(v.y * COV_XXYY);
        Tile2[e + 3u] = Half(v.z * COV_XY);
    }
    GroupMemoryBarrierWithGroupSync();

    bool inside = (int(x) < recon.x) && (int(y) < recon.y);
    float3 outA[2];
    float3 outB[2];
    [unroll]
    for (uint r = 0u; r < 2u; ++r)
    {
        outA[r] = float3(0.0f, 0.0f, 0.0f);
        outB[r] = outA[r];
    }

    if (inside)
    {
        float posx = Fma(float(x) + 0.5f, scaleX, offsetX);
        float posy[2];
        int bx = int(floor(posx - 0.5f));
        int by[2];
        float dx0 = Half((float(bx) + 0.5f) - posx);
        float dy0[2];
        [unroll]
        for (uint r = 0u; r < 2u; ++r)
        {
            posy[r] = Fma(float(y + r) + 0.5f, scaleY, offsetY);
            by[r] = int(floor(posy[r] - 0.5f));
            dy0[r] = Half((float(by[r]) + 0.5f) - posy[r]);
        }
        // the odd row keeps the even row's three texel rows (PTX 301-332)
        if (by[1] != by[0])
            dy0[1] = Half(dy0[1] - 1.0f);

        // --- the 3x3 Gaussians, f16 accumulation ----------------------------
        float acc0[2][4], acc1[2][4], acc2[2][4];
        float weight[2];
        [unroll]
        for (uint r = 0u; r < 2u; ++r)
        {
            weight[r] = 0.0f;
            [unroll]
            for (uint c = 0u; c < 4u; ++c)
            {
                acc0[r][c] = 0.0f;
                acc1[r][c] = 0.0f;
                acc2[r][c] = 0.0f;
            }
        }

        [unroll]
        for (uint i = 0u; i < 3u; ++i)
        {
            float dxi = Half(dx0 + float(int(i) - 1));
            float dx2 = Half(dxi * dxi);
            int col = bx - origin.x + 1 + int(i);
            [unroll]
            for (uint j = 0u; j < 3u; ++j)
            {
                int row = by[0] - origin.y + 1 + int(j);
                uint e = uint(row * 10 + col) * 4u;
                float t0[4], t1[4], t2[4];
                [unroll]
                for (uint c = 0u; c < 4u; ++c)
                {
                    t0[c] = Tile0[e + c];
                    t1[c] = Tile1[e + c];
                    t2[c] = Tile2[e + c];
                }
                [unroll]
                for (uint r = 0u; r < 2u; ++r)
                {
                    float dy = Half(dy0[r] + float(int(j) - 1));
                    float dy2 = Half(dy * dy);
                    float dxy = Half(dxi * dy);
                    float ex = Half(Half(dx2 * t2[1]) + Half(dy2 * t2[2]));
                    ex = Half(ex + Half(dxy * t2[3]));
                    float w = exp2(ex);
                    float wh = Half(w);
                    [unroll]
                    for (uint c = 0u; c < 4u; ++c)
                    {
                        acc0[r][c] = F16Fma(wh, t0[c], acc0[r][c]);
                        acc1[r][c] = F16Fma(wh, t1[c], acc1[r][c]);
                        acc2[r][c] = F16Fma(wh, t2[c], acc2[r][c]);
                    }
                    weight[r] = weight[r] + w;
                }
            }
        }

        // --- candidates and blend -------------------------------------------
        [unroll]
        for (uint r = 0u; r < 2u; ++r)
        {
            float inv = rcp(weight[r]);
            float3 cur = float3(acc0[r][0], acc0[r][1], acc0[r][2]) * inv;
            float3 mixv = float3(acc0[r][3], acc1[r][0], acc1[r][1]) * inv;
            float3 gatev = float3(acc1[r][2], acc1[r][3], acc2[r][0]) * inv;

            // The display-motion fetch is not performed here: with `reset != 0`
            // the history sample is the colour texel regardless of where the
            // motion points, so the vector cannot reach any output.  It also
            // samples `trunc(pos/recon*display)` - up to (1919, 1079) - while the
            // bound texture is 960x540, so issuing it reads out of bounds and,
            // on this driver, makes the dispatch non-deterministic (EXP-018 §2).
            // It must be restored, together with the Catmull-Rom path, when the
            // history branch is implemented.

            // the colour texel at floor(pos), clamped
            int2 near = FoldColour(int2(int(floor(posx)), int(floor(posy[r]))));
            float3 fallback = Colour.Load(int3(near, 0)).rgb;

            // the three weights, sorted; equal vectors mean the samples coincide
            float lo = min(mixv.x, mixv.y);
            float hi = max(mixv.x, mixv.y);
            float med = lo < mixv.z ? mixv.z : lo;
            float big = max(hi, med);
            float second = min(hi, med);
            float k1 = clamp(Fma(big, C_1_1, C_M0_05), 0.0f, 1.0f);
            float k2 = clamp(Fma(second, C_1_1, C_M0_05), 0.0f, 1.0f);
            float den = max(k1 + k2, RC_FLOOR);
            float kn = rcp(den);
            float b2 = k2 * kn;
            float b1 = k1 * kn;
            if (b2 < INV_255)
            {
                b2 = 0.0f;
                b1 = 1.0f;
            }

            // `reset` selects the colour texel; that is the only path this frame
            // takes, and the fixture rejects reset != 0
            float3 h1 = fallback;
            float3 h2 = (b2 > 0.0f) ? h1 : float3(0.0f, 0.0f, 0.0f);
            // f525 = b2*h2 (rounded), then fma(b1, h1, f525)
            float3 was; 
            was.x = b2 * h2.x;
            was.y = b2 * h2.y;
            was.z = b2 * h2.z;
            float3 hist;
            hist.x = Fma(b1, h1.x, was.x);
            hist.y = Fma(b1, h1.y, was.y);
            hist.z = Fma(b1, h1.z, was.z);

            float g0 = clamp((gain * weight[r]) * gatev.x, 0.0f, 1.0f);
            float ga = Fma(g0, 1.0f - gatev.y, gatev.y);
            float gb = Fma(g0, 1.0f - gatev.z, gatev.z);

            float3 a, b;
            a.x = Fma(cur.x - hist.x, ga, hist.x);
            a.y = Fma(cur.y - hist.y, ga, hist.y);
            a.z = Fma(cur.z - hist.z, ga, hist.z);
            b.x = Fma(cur.x - hist.x, gb, hist.x);
            b.y = Fma(cur.y - hist.y, gb, hist.y);
            b.z = Fma(cur.z - hist.z, gb, hist.z);

            bool badA = any(isnan(a)) || any(isinf(a));
            bool badB = any(isnan(b)) || any(isinf(b));
            outA[r] = badA ? cur : a;
            outB[r] = badB ? cur : b;
        }
    }

    if (inside)
    {
        [unroll]
        for (uint r = 0u; r < 2u; ++r)
        {
            uint px = x;
            uint py = y + r;
            // outA -> RGBA16F bytes, truncating, two words per pixel
            uint hr = F16Trunc(outA[r].x);
            uint hg = F16Trunc(outA[r].y);
            uint hb = F16Trunc(outA[r].z);
            History[uint2(2u * px, py)] = hr | (hg << 16);
            History[uint2(2u * px + 1u, py)] = hb;
            // outB -> the packed 11-11-10 word
            uint cr = uint(clamp(floor(Fma(outB[r].x, 2047.0f, 0.5f)), 0.0f, 2047.0f));
            uint cg = uint(clamp(floor(Fma(outB[r].y, 2047.0f, 0.5f)), 0.0f, 2047.0f));
            uint cb = uint(clamp(floor(Fma(outB[r].z, 1023.0f, 0.5f)), 0.0f, 1023.0f));
            Packed[py * uint(recon.x) + px] = cr | (cg << 11) | (cb << 22);
        }
    }

    // --- the half-resolution mean -------------------------------------------
    [unroll]
    for (uint c = 0u; c < 3u; ++c)
        ColumnSum[lane * 3u + c] = inside ? (outA[0][c] + outA[1][c]) : 0.0f;
    GroupMemoryBarrierWithGroupSync();

    if (inside && (x & 1u) == 0u && (x ^ 1u) < uint(recon.x))
    {
        uint other = (lane ^ 1u) * 3u;
        float3 mean;
        [unroll]
        for (uint c = 0u; c < 3u; ++c)
            mean[c] = (ColumnSum[lane * 3u + c] + ColumnSum[other + c]) * 0.25f;
        float4 q;
        q.x = Unorm10Value(Unorm10(mean.x));
        q.y = Unorm10Value(Unorm10(mean.y));
        q.z = Unorm10Value(Unorm10(mean.z));
        q.w = 0.0f;
        HalfOut[uint2(x / 2u, y / 2u)] = q;
    }
}
