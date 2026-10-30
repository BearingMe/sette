// M09 I3, second dispatch: the INT8 Q projection itself.
//
//   in   Codes   int8 [tokens][K], four per word     (this token's activations)
//        WCodes  int8 [N][K], four per word          (K-major per output column)
//        AScale  f32 per token
//        WScale  f32 per output column
//   out  Y       f16 pairs, [tokens][N] at tok*N + n
//
// This writes exactly the resource the FP16 `Matmul` writes, in the same layout
// and format, so `attention` is unchanged: it reads `q` as f16 and re-quantises
// it to E4M3 itself (`Q8`).  The only arithmetic differences from the FP16 path
// are (a) both operands are INT8, (b) the reduction is exact int32 over the whole
// K instead of one f16 rounding per 32 products, and (c) the rescale is a single
// f32 multiply pair.  EXP-023 separates those contributions.
//
// `dot4add_i8packed` is RDNA 3's `v_dot4_i32_i8`: four signed 8-bit products and
// a 32-bit accumulate in one instruction.  K = 64 needs 16 of them per output
// element, and |sum| <= 64*127*127 = 1 032 256, so int32 is exact and
// `(float)acc` converts without loss.

#include "swin_q8_common.hlsli"

cbuffer Q8Params : register(b0)
{
    uint Tokens;
    uint K;
    uint N;
    uint Unused;
};

Buffer<uint> Codes : register(t0);
Buffer<uint> WCodes : register(t1);
Buffer<uint> AScale : register(t2);
Buffer<uint> WScale : register(t3);
RWBuffer<uint> Y : register(u0);

[numthreads(64, 1, 1)]
void Matmul(uint3 id : SV_DispatchThreadID)
{
    uint tiles = N >> 4;
    uint tile = id.x % tiles, tok = id.x / tiles;
    if (tok >= Tokens)
        return;
    uint n0 = tile * 16u;
    uint groups = K >> 2;
    uint aBase = tok * groups;
    float as = asfloat(AScale[tok]);

    int acc[16];
    for (uint j = 0; j < 16u; ++j)
        acc[j] = 0;
    for (uint g = 0; g < groups; ++g)
    {
        uint a = Codes[aBase + g];
        for (uint j = 0; j < 16u; ++j)
            acc[j] = dot4add_i8packed(a, WCodes[(n0 + j) * groups + g], acc[j]);
    }
    uint outBase = tok * N;
    for (uint j = 0; j < 16u; j += 2u)
    {
        uint n = n0 + j;
        float y0 = (float)acc[j] * as * asfloat(WScale[n]);
        float y1 = (float)acc[j + 1u] * as * asfloat(WScale[n + 1u]);
        Y[(outBase + n) >> 1] = ToHalfBits(y0) | (ToHalfBits(y1) << 16);
    }
}
