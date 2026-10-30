// M09 I3, first dispatch: per-token INT8 encoding of the Q operand.
//
//   in   A      f16 pairs, row-major [tokens][K]   (norm1's output, GPU-resident)
//   out  Codes  int8, four per word, [tokens][K]
//        Scales f32 per token, `max_abs * (1/127)` in f32
//
// The encoding is the contract M07/M09 validated (`m_int8.quantize_rows`):
// symmetric signed INT8, no zero point, `inv = 127/max_abs` in f32, codes are
// `round_half_even(v * inv)` clamped to +-127, and a row whose maximum is zero
// uses `max_abs = 1` (codes 0, scale 1/127) instead of dividing by zero.
//
// The product `v * inv` is formed in f32 because that is what the kernel can do;
// the CPU reference reproduces it (EXP-023 measures the codes this changes
// against `m_int8`'s f64 ratio rather than assuming there are none).

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

[numthreads(64, 1, 1)]
void Quantize(uint3 id : SV_DispatchThreadID)
{
    uint tok = id.x;
    if (tok >= Tokens)
        return;
    uint base = tok * K;
    float mx = 0.0f;
    for (uint c = 0; c < K; ++c)
        mx = max(mx, abs(HalfAt(A, base + c)));
    if (mx == 0.0f)
        mx = 1.0f;
    // 127/max_abs, correctly rounded to f32.
    //
    // `127.0f / mx` lowers to v_rcp_f32 + v_mul_f32 (measured in EXP-023), which
    // is within about one ulp of the exact quotient but not equal to it: on the
    // captured enc1 operand that moved 4 382 of 8 744 960 codes (0.05%, one of
    // the 136 640 tokens' inverses differing by 1 ulp affects a whole row).  The
    // correction below makes the quotient exact to f64 and then rounds once to
    // f32: `(double)invf * mx` is exact (24+24 significant bits), so `r` is the
    // exact residual of the candidate, and adding `r/mx` leaves an error of
    // order 1e-16 relative - far below the f32 rounding boundary.  A f64
    // division for the quotient itself was tried and rejected: the device's f64
    // divider is one ulp low for some divisors (mx = 2.25 yields an inverse whose
    // product is 63.49999999999999 where the exact product is 63.5).
    float invf = 127.0f / mx;
    double residual = 127.0 - (double)invf * (double)mx;
    float inv = (float)((double)invf + residual / (double)mx);
    Scales[tok] = asuint(mx * 0.0078740157f);       // f32(1/127), the epilogue scale

    uint wordBase = base >> 2;
    for (uint g = 0; g < (K >> 2); ++g)
    {
        uint word = 0;
        for (uint j = 0; j < 4u; ++j)
        {
            float v = HalfAt(A, base + 4u * g + j);
            word |= (uint)(QuantCode(v * inv) & 0xFF) << (8u * j);
        }
        Codes[wordBase + g] = word;
    }
}




