// EXP-027: materialize the existing GeluQ, with no matrix arithmetic.
#include "swin_layer_cs.hlsl"
[numthreads(64, 1, 1)]
void Gelu(uint3 id : SV_DispatchThreadID)
{
    uint tok = id.x;
    if (tok >= pTokens) return;
    uint width = pNC * 32u;
    for (uint c = 0; c < width; c += 2u)
    {
        uint i = tok * width + c;
        Store2(U0, i, GeluQ(HalfAt(B0, i)), GeluQ(HalfAt(B0, i + 1u)));
    }
}

// Validation only: original eight FP16 down steps on the same materialized G.
[numthreads(64, 1, 1)]
void DownFP16(uint3 id : SV_DispatchThreadID)
{
    uint tok = id.x;
    if (tok >= pTokens) return;
    for (uint c = 0; c < pC; c += 2u)
    {
        float x0 = HalfAt(B2, tok*pC+c), x1 = HalfAt(B2, tok*pC+c+1u);
        for (uint chunk = 0; chunk < pNC; ++chunk)
        {
            double sum0 = 0.0, sum1 = 0.0;
            for (uint j = 0; j < 32u; ++j)
            {
                double g = (double)HalfAt(B0, tok*pNC*32u+chunk*32u+j);
                uint w = pOffW2+chunk*32u*pC+j*pC+c;
                sum0 += g*(double)asfloat(B1[w]);
                sum1 += g*(double)asfloat(B1[w+1u]);
            }
            x0 = R16D(sum0+(double)x0); x1 = R16D(sum1+(double)x1);
        }
        Store2(U0, tok*pC+c, x0, x1);
    }
}
