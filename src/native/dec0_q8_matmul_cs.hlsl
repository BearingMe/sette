// EXP-024: same packed signed dot4 reduction as gemm_int8_cs.hlsl.
// Offline weights are decoded from the physical blocked blob; activations are
// encoded on GPU directly from the two E4M3 planes. Bias is added once in f32.
#include "swin_q8_common.hlsli"
Buffer<uint> A : register(t0);
Buffer<uint> W : register(t1);
Buffer<uint> AS : register(t2);
Buffer<uint> WS : register(t3);
Buffer<uint> Bias : register(t4);
RWBuffer<uint> HalfOut : register(u0);
RWBuffer<uint> AccOut : register(u1);
RWBuffer<uint> FloatOut : register(u2);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint pair = id.x + id.y * 130560u;
    if (pair >= 130560u * 40u) return;
    uint row = pair / 40u, n = (pair % 40u) * 2u;
    int acc0 = 0, acc1 = 0;
    for (uint g = 0; g < 16u; ++g)
    {
        uint a = A[row*16u+g];
        acc0 = dot4add_i8packed(a, W[n*16u+g], acc0);
        acc1 = dot4add_i8packed(a, W[(n+1u)*16u+g], acc1);
    }
    precise float y0 = float(acc0) * asfloat(AS[row]) * asfloat(WS[n]);
    precise float y1 = float(acc1) * asfloat(AS[row]) * asfloat(WS[n+1u]);
    y0 = y0 + asfloat(Bias[n]);
    y1 = y1 + asfloat(Bias[n+1u]);
    uint i = pair*2u;
    AccOut[i] = asuint(acc0); AccOut[i+1u] = asuint(acc1);
    FloatOut[i] = asuint(y0); FloatOut[i+1u] = asuint(y1);
    HalfOut[pair] = ToHalfBits(y0) | (ToHalfBits(y1) << 16);
}
