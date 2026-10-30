// M09 I3: shared helpers for the INT8 Q projection (`swin_q8_*_cs.hlsl`).
//
// `swin_layer_cs.hlsl` keeps its own copies of the f16 conversion because that
// file is the measured FP16 contract and is not touched by the INT8 experiment;
// every value produced here is compared against it bit for bit (EXP-023), so a
// divergence in either copy fails the experiment rather than passing silently.

#ifndef SWIN_Q8_COMMON_HLSLI
#define SWIN_Q8_COMMON_HLSLI

float HalfAt(Buffer<uint> b, uint i)
{
    uint w = b[i >> 1];
    return f16tof32((i & 1u) ? (w >> 16) : (w & 0xFFFFu));
}

// Explicit round-to-nearest-even float32 -> float16.  EXP-010 measured the
// driver's own f32->f16 conversion truncating toward zero, so the layer shader
// does this in integer arithmetic and so does this one.
uint ToHalfBits(float value)
{
    uint u = asuint(value);
    uint sign = (u >> 16) & 0x8000u;
    int expo = int((u >> 23) & 0xFFu) - 127 + 15;
    uint mant = u & 0x7FFFFFu;
    if (expo <= 0)
    {
        int shift = 14 - expo;
        if (shift > 24)
            return sign;
        uint full = (mant | 0x800000u) >> shift;
        uint rest = (mant | 0x800000u) & ((1u << shift) - 1u);
        uint half = 1u << (shift - 1);
        if (rest > half || (rest == half && (full & 1u)))
            full += 1u;
        return sign | full;
    }
    if (expo >= 31)
        return sign | 0x7C00u;
    uint h = (uint(expo) << 10) | (mant >> 13);
    uint rest = mant & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u)))
        h += 1u;
    return sign | h;
}

// f32 ratio -> INT8 code: round half to even, then saturate.
//
// `round()` is DXIL's Round, which the spec defines as "nearest integer, half
// way cases to the nearest even integer" - the same rule as the `np.rint` that
// `m_int8` uses.  An explicit compare-and-select version of the same rule was
// tried first and did not survive code generation: on an exact tie it produced
// the odd side (63 for 63.5 where np.rint gives 64), so the rule is left to the
// instruction that defines it.  EXP-023 records both forms.
int QuantCode(float r)
{
    return clamp((int)round(r), -127, 127);
}

#endif  // SWIN_Q8_COMMON_HLSLI
