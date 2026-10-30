// EXP-026 validation only: materialize the original fused W1+b1 boundary.
#include "swin_layer_cs.hlsl"
[numthreads(64, 1, 1)]
void Up(uint3 id : SV_DispatchThreadID)
{
    uint chunk = id.x % pNC, tok = id.x / pNC;
    if (tok >= pTokens) return;
    float z[32];
    for (uint j = 0; j < 32u; ++j) z[j] = asfloat(B1[pOffB1 + chunk * 32u + j]);
    for (uint k0 = 0; k0 < pC; k0 += 32u)
    {
        double sum[32];
        for (uint j = 0; j < 32u; ++j) sum[j] = 0.0;
        for (uint k = 0; k < 32u; ++k)
        {
            double a = (double)HalfAt(B0, tok * pC + k0 + k);
            for (uint j = 0; j < 32u; ++j)
                sum[j] += a * (double)asfloat(B1[pOffW1 + chunk*pC*32u + (k0+k)*32u+j]);
        }
        for (uint j = 0; j < 32u; ++j) z[j] = R16D(sum[j] + (double)z[j]);
    }
    for (uint j = 0; j < 32u; j += 2u)
        Store2(U0, tok*pNC*32u + chunk*32u+j, z[j], z[j+1u]);
}
