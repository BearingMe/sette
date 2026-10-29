// M05 B4.2: native reconstruction of `rrlite_downsample_kernel_static_ldr`
// (Preset M, mvhi / LDR / static, the final 3 -> 2 output resample).
//
// Source: d4r `engine/down_m.comp` plus the captured launch's parameter block.
// The captured launch fixes four things the source alone leaves open:
//
//   * the block is 16x16 over the *output* image (120x68 groups), not 8x64.  The
//     filter is tile-independent - the tile only shares intermediate rows - so
//     this shader computes each output pixel directly and skips the shared
//     memory tile while producing the same halves;
//   * the packed source is `(2880,1620)`, the output origin zero, the colour
//     window `(0,0)-(960,540)`, `out2rnd = (0.5,0.5)`, `rndOffs = 0`, exposure
//     0 and grid.z = 1.  `out2rnd` is the colour/output scale, not 1.5: the
//     colour texture is read at render resolution only to fetch alpha;
//   * the variant is `_ldr_`, so a packed word is display-referred colour and
//     the HDR branch (range expansion, clamp to [1e-4,3], exposure division)
//     does not run;
//   * `NumWorkGroups.z` is 1, which is the constant d4r's parity expression
//     `(q + (NumWorkGroups.z - 1)) & 1` folds to.
//
// Geometry.  Cancelling the tile terms leaves an output pixel (px,py) reading
// the source at
//     horizontal centre  A = ceil(1.5 px)          (d4r: base.x + 3*x0/2 + o)
//     vertical centre    T = floor(1.5 (py + 0.5))  (d4r: base.y + c.y)
// with taps A-2..A+2 and T-2..T+2, each clamped to the source rectangle, and the
// weights mirrored by the parity of the output coordinate.  The resulting
// sample sits at 1.5 px + 0.29 / 1.5 px + 0.21 in source pixels, i.e. within
// 0.05 pixel of the exact 3:2 grid position 1.5 px + 0.25, which is why the
// stored weights are self-consistent (they are NGX's binary16 constants).
//
// Arithmetic.  The stage is f16 end to end: `unpack` yields binary16, the
// horizontal pass stores halves, `filter5` is one f16 multiply plus four f16
// fused multiply-adds and a clamp.  A fused f16 multiply-add rounds once; f32
// arithmetic would round twice (to f32, then to f16), and native `half` - which
// this compiler lowers to a divide-and-add pair plus a *truncating* f32->f16
// conversion (EXP-010) - is no better.  So every f16 operation is reproduced
// exactly with three facts:
//
//   1. a product of two f16 values has a 22-bit significand, so it is exact in
//      f32 and in the f32 range;
//   2. TwoSum returns the exact sum of two f32 values as (high, low);
//   3. rounding an exact pair to f16 only needs the high limb's position
//      relative to the f16 midpoint (which is an exact multiple of the f32 ulp)
//      and, when the high limb *is* on a midpoint, the sign of the low limb.
//
// `RoundPair` therefore rounds to nearest-even exactly, and `HalfFma` is a true
// fused f16 multiply-add.  No f64 is involved.
//
// Validation: `scripts/m_stage_fixture.py` compares the result with the captured
// surface byte for byte, and `tests/test_downsample_contract.py` enumerates the
// whole unpacked code space (EXP-012).

cbuffer Params : register(b0)
{
    int4 srcSize_outOrigin;    // packed source width/height, output origin
    int4 colOrigin_colSize;    // colour window origin, colour window size
    float4 out2rnd_rndOffs;    // output -> colour scale, colour offset
    float4 expo_unused;        // exposure: only the HDR variant reads it
};

Buffer<uint> Src : register(t0);
Texture2D<float4> ColourTex : register(t1);
RWTexture2D<float4> OutImg : register(u0);

// d4r's five taps as binary16 bit patterns: NGX rounds its f32 constants to the
// nearest half (cvt.rn), while glslang folds them toward zero - W0 became
// -0.041321 instead of -0.041351, which moved 6% of the reference frame's
// outputs by one code or more.  The values are -0.041347627, 0.087289415,
// 0.6202144, 0.37212864, -0.038284853.
static const float kTap[5] = {
    f16tof32(0xA94Bu), f16tof32(0x2D96u), f16tof32(0x38F6u), f16tof32(0x35F4u),
    f16tof32(0xA8E7u)
};

// 2^e as an exact power-of-two scale.
float Pow2(int e)
{
    return asfloat(uint(e + 127) << 23);
}

// Exact a + b as a pair: `high + low == a + b`.  Knuth's branchless TwoSum.
//
// Every step is `precise`: without it the compiler contracts the caller's
// `w * b` into this add as a fused multiply-add, which is a *different* (more
// accurate) value, and the residual then comes out with the wrong sign - 51 of
// the 8.3M halves in the reference frame landed one code low in exactly the tie
// cases before this was pinned down.
float2 TwoSum(float a, float b)
{
    precise float sum = a + b;
    precise float bb = sum - a;
    precise float ea = a - (sum - bb);
    precise float eb = b - bb;
    return float2(sum, ea + eb);
}

// Round the exact value `high + low` (|low| <= ulp(high)/2) to nearest-even
// binary16, returned as the f16 value widened to f32.  The magnitude is scaled
// so the f16 last bit sits at 2^0, which makes the f16 midpoint exactly 0.5; the
// sign is applied at the end, so the integer part is never negative.
float RoundPair(float high, float low)
{
    uint u = asuint(high);
    uint sign = (u >> 16) & 0x8000u;          // the half's sign bit
    uint magnitude = u & 0x7FFFFFFFu;
    if (magnitude == 0u || magnitude >= 0x7F800000u)
        return high;                          // zero, infinity and NaN pass through
    float value = sign ? -high : high;        // |high|
    float residual = sign ? -low : low;       // the low limb, oriented with |high|
    int exponent = int(magnitude >> 23) - 127;
    int quantum = max(exponent - 10, -24);    // exponent of the half's last bit
    float scaled = value * Pow2(-quantum);
    float whole = floor(scaled);
    float rest = scaled - whole;
    int n = int(whole);
    // 0.5 is exactly representable and `rest` is an exact multiple of the f32
    // ulp here, so this is a real tie test; the low limb can only break a tie.
    if (rest > 0.5f || (rest == 0.5f && (residual > 0.0f
                                         || (residual == 0.0f && (n & 1) != 0))))
        n += 1;
    if (exponent <= -15)
        return f16tof32(sign | uint(min(n, 1024)));   // subnormal, or the smallest normal
    if (n == 2048) {                          // rounding carried into the next binade
        n = 1024;
        exponent += 1;
    }
    if (exponent > 15)
        return f16tof32(sign | 0x7C00u);      // overflow, unreachable for these operands
    return f16tof32(sign | (uint(exponent + 15) << 10) | uint(n - 1024));
}

// Nearest binary16 to a single exact f32 value.
float Half(float value)
{
    return RoundPair(value, 0.0f);
}

float3 Half3(float3 value)
{
    return float3(Half(value.x), Half(value.y), Half(value.z));
}

// A fused f16 multiply-add: rn16(w * b + c).  `w * b` is exact in f32 (fact 1
// above), so the only rounding is the one RoundPair performs - provided the
// product is not contracted into the sum, hence `precise`.
float HalfFma(float w, float b, float c)
{
    precise float product = w * b;
    precise float2 sum = TwoSum(product, c);
    return RoundPair(sum.x, sum.y);
}

// LDR unpack of one packed word: 11-11-10 bits, each normalised by its full
// code range.  The f32 division is exact to the f16 grid for every code in the
// range (the whole 2048/1024-code spaces are enumerated by the contract test).
float UnpackLdr(uint word, int channel)
{
    if (channel == 0)
        return Half(float(word & 2047u) / 2047.0f);
    if (channel == 1)
        return Half(float((word >> 11) & 2047u) / 2047.0f);
    return Half(float(word >> 22) / 1023.0f);
}

// d4r's filter5: an f16 multiply, four fused f16 multiply-adds, then a clamp to
// the range of the taps so the resample cannot ring.
float3 Filter(float3 t0, float3 t1, float3 t2, float3 t3, float3 t4, bool odd)
{
    float w0 = kTap[odd ? 4 : 0], w1 = kTap[odd ? 3 : 1], w2 = kTap[2];
    float w3 = kTap[odd ? 1 : 3], w4 = kTap[odd ? 0 : 4];
    float3 s = Half3(w0 * t0);
    s = float3(HalfFma(w1, t1.x, s.x), HalfFma(w1, t1.y, s.y), HalfFma(w1, t1.z, s.z));
    s = float3(HalfFma(w2, t2.x, s.x), HalfFma(w2, t2.y, s.y), HalfFma(w2, t2.z, s.z));
    s = float3(HalfFma(w3, t3.x, s.x), HalfFma(w3, t3.y, s.y), HalfFma(w3, t3.z, s.z));
    s = float3(HalfFma(w4, t4.x, s.x), HalfFma(w4, t4.y, s.y), HalfFma(w4, t4.z, s.z));
    float3 lo = min(min(min(t0, t1), min(t2, t3)), t4);
    float3 hi = max(max(max(t0, t1), max(t2, t3)), t4);
    return clamp(s, lo, hi);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    OutImg.GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
        return;
    const int2 pix = int2(id.xy);
    const int srcW = srcSize_outOrigin.x, srcH = srcSize_outOrigin.y;
    // d4r's bounds check: the borrowed image may extend past the true output
    // rectangle, and srcSize = output*3/2 makes it exact for even extents.
    if (3 * pix.x >= 2 * srcW || 3 * pix.y >= 2 * srcH)
        return;

    const int centreX = (3 * pix.x + 1) >> 1;              // ceil(1.5 px)
    const int centreY = (6 * pix.y + 3) >> 2;              // floor(1.5 (py + 0.5))
    const bool oddX = (pix.x & 1) != 0;
    const bool oddY = (pix.y & 1) != 0;

    // The horizontal pass: one filtered source column per source row, run once
    // per row the vertical pass needs.  Same arithmetic and the same halves as
    // d4r's tiled version, which shares them across the block.
    float3 rows[5];
    [unroll]
    for (int t = 0; t < 5; ++t) {
        const uint row = uint(clamp(centreY + t - 2, 0, srcH - 1));
        float3 tap[5];
        [unroll]
        for (int k = 0; k < 5; ++k) {
            const uint word = Src[row * uint(srcW) + uint(clamp(centreX + k - 2, 0, srcW - 1))];
            tap[k] = float3(UnpackLdr(word, 0), UnpackLdr(word, 1), UnpackLdr(word, 2));
        }
        rows[t] = Filter(tap[0], tap[1], tap[2], tap[3], tap[4], oddX);
    }
    const float3 lin = Filter(rows[0], rows[1], rows[2], rows[3], rows[4], oddY);

    // Alpha is not reconstructed: it is the render-resolution colour's alpha at
    // the output pixel's own centre, mapped back into the colour window.
    const int2 colSize = colOrigin_colSize.zw;
    const int2 window = clamp(
        int2(floor((float2(pix) + 0.5f) * out2rnd_rndOffs.xy + out2rnd_rndOffs.zw)),
        int2(0, 0), colSize - 1) + colOrigin_colSize.xy;
    const float alpha = ColourTex.Load(int3(window, 0)).a;

    // The captured variant is the LDR one, so the resample *is* the display
    // value; `finalOutputValue` is the identity for an rgba16f target.
    OutImg[pix] = float4(lin, alpha);
}
