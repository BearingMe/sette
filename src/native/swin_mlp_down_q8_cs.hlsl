// EXP-027: full K=256 signed INT8 down projection, plus original f16 residual.
// One exact INT32 reduction, f32 rescale pair, f32 residual add, f16 RNE store.
#include "swin_q8_common.hlsli"
cbuffer Q8Params : register(b0) { uint Tokens, K, N, Unused; };
Buffer<uint> A : register(t0);
Buffer<uint> W : register(t1);
Buffer<uint> AS : register(t2);
Buffer<uint> WS : register(t3);
Buffer<uint> Residual : register(t4);
RWBuffer<uint> Y : register(u0);
RWBuffer<uint> Acc : register(u1);
RWBuffer<uint> FloatOut : register(u2);
[numthreads(64, 1, 1)]
void Down(uint3 id : SV_DispatchThreadID)
{
    uint tile = id.x % (N/16u), tok = id.x / (N/16u);
    if (tok >= Tokens) return;
    int acc[16];
    for (uint j = 0; j < 16u; ++j) acc[j] = 0;
    for (uint g = 0; g < K/4u; ++g)
    {
        uint a = A[tok*(K/4u)+g];
        for (uint j = 0; j < 16u; ++j)
            acc[j] = dot4add_i8packed(a, W[(tile*16u+j)*(K/4u)+g], acc[j]);
    }
    for (uint j = 0; j < 16u; j += 2u)
    {
        uint n = tile*16u+j, index = tok*N+n;
        precise float y0 = float(acc[j])*asfloat(AS[tok])*asfloat(WS[n]);
        precise float y1 = float(acc[j+1u])*asfloat(AS[tok])*asfloat(WS[n+1u]);
        y0 = y0 + HalfAt(Residual, index);
        y1 = y1 + HalfAt(Residual, index+1u);
        Y[index/2u] = ToHalfBits(y0) | (ToHalfBits(y1)<<16);
        Acc[index] = asuint(acc[j]); Acc[index+1u] = asuint(acc[j+1u]);
        FloatOut[index] = asuint(y0); FloatOut[index+1u] = asuint(y1);
    }
}
